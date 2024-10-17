import os
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
from utils.dataset import load_mnist, load_fmnist, denorm, select_from_dataset
from utils.wgan import compute_gradient_penalty
from networks.CAQCC import CAQ_CC
from networks.CAQCQ import CAQ_QC
import pandas as pd
from scipy.linalg import sqrtm
import torch.nn as nn
import torch.nn.functional as F


# FID score calculation
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# Noise upper bound adjustment using Leaky ReLU and Sigmoid function
def get_noise_upper_bound(gen_loss, disc_loss, original_ratio):
    """
    Adjusts noise upper bound based on the ratio of generator loss to discriminator loss.
    - gen_loss: Current generator loss (L_G)
    - disc_loss: Current discriminator loss (L_C)
    - original_ratio: The ratio of initial generator loss to discriminator loss (L_G1/L_C1)

    Returns:
        Noise upper bound which is limited between π/6 and 2π/3.
    """
    current_ratio = disc_loss.detach().cpu().numpy() / gen_loss.detach().cpu().numpy()
    sigmoid_value = torch.sigmoid(torch.tensor(current_ratio - original_ratio))
    leaky_relu_value = F.leaky_relu(sigmoid_value, negative_slope=0.01)
    noise_max = (2 * math.pi / 3) * leaky_relu_value.item()
    noise_max = max(noise_max, math.pi / 6)

    return noise_max


# Training function for CAQ model
def train_CAQ(classes_str, dataset_str, patches, layers, n_data_qubits, batch_size, out_folder, checkpoint, randn,
              patch_shape, qcritic):
    classes = list(set([int(digit) for digit in classes_str]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 100
    image_size = 28
    channels = 1

    if dataset_str == "mnist":
        dataset = select_from_dataset(load_mnist(image_size=image_size), 1500, classes)
    elif dataset_str == "fmnist":
        dataset = select_from_dataset(load_fmnist(image_size=image_size), 1500, classes)

    ancillas = 1
    if n_data_qubits:
        qubits = n_data_qubits + ancillas
    else:
        qubits = math.ceil(math.log(image_size ** 2 // patches, 2)) + ancillas

    lr_D = 0.005 if qcritic else 0.0001
    lr_G = 0.0002
    b1 = 0.5
    b2 = 0.999
    latent_dim = qubits
    lambda_gp = 10
    n_critic = 5
    sample_interval = 200

    out_dir = f"{out_folder}/{classes_str}_{patches}p_{layers}l_{batch_size}bs"
    if randn:
        out_dir += "_randn"
    if patch_shape[0] and patch_shape[1]:
        out_dir += f"_{patch_shape[0]}x{patch_shape[1]}ps"
    os.makedirs(out_dir, exist_ok=True)

    if qcritic:
        gan = CAQ_QC(image_size=image_size, channels=channels, n_generators=patches, n_gen_qubits=qubits,
                     n_ancillas=ancillas, n_gen_layers=layers, patch_shape=patch_shape, n_critic_qubits=10,
                     n_critic_layers=150)
    else:
        gan = CAQ_CC(image_size=image_size, channels=channels, n_generators=patches, n_qubits=qubits,
                     n_ancillas=ancillas, n_layers=layers, patch_shape=patch_shape)

    critic = gan.critic.to(device)
    generator = gan.generator.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    optimizer_C = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))

    fixed_z = torch.randn(batch_size, latent_dim, device=device) if randn else torch.rand(batch_size, latent_dim, device=device)

    wasserstein_distance_history = []
    saved_initial = False
    batches_done = 0
    original_ratio = None
    noise_upper_bound = math.pi / 6

    if checkpoint != 0:
        critic.load_state_dict(torch.load(out_dir + f"/critic-{checkpoint}.pt"))
        generator.load_state_dict(torch.load(out_dir + f"/generator-{checkpoint}.pt"))
        wasserstein_distance_history = list(np.load(out_dir + "/wasserstein_distance.npy"))
        saved_initial = True
        batches_done = checkpoint

    for epoch in range(n_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            if not saved_initial:
                fixed_images = generator(fixed_z.to(device))
                save_image(denorm(fixed_images), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(real_images), os.path.join(out_dir, 'real_samples.png'), nrow=5)
                saved_initial = True

            real_images = real_images.to(device)
            optimizer_C.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device) * noise_upper_bound if randn else torch.rand(batch_size, latent_dim, device=device) * noise_upper_bound
            fake_images = generator(z)

            real_validity = critic(real_images)
            fake_validity = critic(fake_images)
            gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images, device)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
            wasserstein_distance_history.append(wasserstein_distance.item())

            d_loss.backward()
            optimizer_C.step()

            optimizer_G.zero_grad()

            if i % n_critic == 0:
                fake_images = generator(z)
                fake_validity = critic(fake_images)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()

                if original_ratio is None:
                    original_ratio = d_loss.detach().cpu().numpy() / g_loss.detach().cpu().numpy()
                noise_upper_bound = get_noise_upper_bound(g_loss, d_loss, original_ratio)

                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Wasserstein Distance: {wasserstein_distance.item()}]")

                batches_done += n_critic

                if batches_done % sample_interval == 0:
                    fixed_images = generator(fixed_z.to(device))
                    save_image(denorm(fixed_images), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                    torch.save(critic.state_dict(), os.path.join(out_dir, 'critic-{}.pt'.format(batches_done)))
                    torch.save(generator.state_dict(), os.path.join(out_dir, 'generator-{}.pt'.format(batches_done)))
                    print("Saved images and model state")

    np.save(os.path.join(out_dir, 'wasserstein_distance.npy'), wasserstein_distance_history)

    real_images_np = real_images.cpu().numpy().reshape(batch_size, -1)
    fake_images_np = fake_images.cpu().detach().numpy().reshape(batch_size, -1)
    fid_score = calculate_fid(real_images_np, fake_images_np)
    print(f"FID Score: {fid_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cl", "--classes", help="classes to train on", type=str, default=[0, 1])
    parser.add_argument("-d", "--dataset", help="dataset to train on", type=str, default='mnist')
    parser.add_argument("-p", "--patches", help="number of sub-generators", type=int, choices=[1, 2, 4, 7, 14, 28],
                        default=4)
    parser.add_argument("-l", "--layers", help="layers per sub-generators", type=int, default=12)
    parser.add_argument("-q", "--qubits", help="number of data qubits per sub-generator", type=int,
                        default=12)
    parser.add_argument("-b", "--batch_size", help="batch_size", type=int, default=16)
    parser.add_argument("-o", "--out_folder", help="output directory", type=str, default="out")
    parser.add_argument("-c", "--checkpoint", help="checkpoint to load from", type=int, default=0)
    parser.add_argument("-rn", "--randn", help="use normal prior, otherwise use uniform prior", action="store_true")
    parser.add_argument("-ps", "--patch_shape", help="shape of sub-generator output (H, W)", default=[14, 14],
                        type=int, nargs=2)
    parser.add_argument("-qc", "--qcritic", help="use quantum critic", action="store_true")
    args = parser.parse_args()

    train_CAQ(args.classes, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder,
              args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
