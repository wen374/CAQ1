import pennylane as qml
import torch
import torch.nn as nn


class CAQ_QC():
    def __init__(self, image_size, channels, n_generators, n_gen_qubits, n_ancillas, n_gen_layers, patch_shape,
                 n_critic_qubits, n_critic_layers):
        self.image_shape = (channels, image_size, image_size)
        self.critic = self.QuantumCritic(n_critic_qubits, n_critic_layers)
        self.generator = self.QuantumGenerator(n_generators, n_gen_qubits, n_ancillas, n_gen_layers, self.image_shape,
                                               patch_shape)

    class QuantumCritic(nn.Module):
        def __init__(self, n_qubits, n_layers):
            super().__init__()
            self.n_qubits = n_qubits
            self.qnode = qml.QNode(self.circuit, qml.device("default.qubit", wires=n_qubits))
            self.weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}
            self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

        def circuit(self, inputs, weights):
            assert inputs.shape[0] <= 2 ** self.n_qubits, "Need more qubits to encode vector"
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), pad_with=0., normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(wires=0))

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.qlayer(x)
            sign = torch.sign(x)
            x = torch.abs(x) * torch.pi / 2 - 0.001  # prevent inf
            x *= sign
            return torch.tan(x)

    class QuantumGenerator(nn.Module):
        def __init__(self, n_generators, n_qubits, n_ancillas, n_layers, image_shape, patch_shape):
            super().__init__()
            self.n_generators = n_generators
            self.n_qubits = n_qubits
            self.n_ancillas = n_ancillas
            self.n_layers = n_layers
            self.q_device = qml.device("default.qubit", wires=n_qubits)
            self.params = nn.ParameterList(
                [nn.Parameter(torch.rand(n_layers, n_qubits, 3), requires_grad=True) for _ in range(n_generators)]
            )
            self.qnode = qml.QNode(self.circuit, self.q_device, interface="torch")

            self.image_shape = image_shape
            self.patch_shape = patch_shape

        def forward(self, x):
            patch_size = 2 ** (self.n_qubits - self.n_ancillas)
            image_pixels = self.image_shape[2] ** 2
            pixels_per_patch = image_pixels // self.n_generators

            if self.patch_shape[0] * self.patch_shape[1] != pixels_per_patch:
                raise ValueError("Patch shape and patch size don't match!")

            output_images = torch.Tensor(x.size(0), 0)

            for sub_generator_param in self.params:
                patches = torch.Tensor(0, pixels_per_patch)
                for item in x:
                    sub_generator_out = self.partial_trace_and_postprocess(item, sub_generator_param).float().unsqueeze(0)
                    patches = torch.cat((patches, sub_generator_out))
                output_images = torch.cat((output_images, patches), 1)

            final_out = output_images.view(output_images.shape[0], *self.image_shape)
            return final_out

        def circuit(self, latent_vector, weights):
            for i in range(self.n_qubits):
                qml.RY(latent_vector[i], wires=i)

            for i in range(self.n_layers):
                for j in range(self.n_qubits):
                    qml.Rot(*weights[i][j], wires=j)

                for j in range(self.n_qubits - 1):
                    qml.CNOT(wires=[j, j + 1])

            return qml.probs(wires=list(range(self.n_qubits)))

        def partial_trace_and_postprocess(self, latent_vector, weights):
            probs = self.qnode(latent_vector, weights)
            probs_given_ancilla_0 = probs[:2 ** (self.n_qubits - self.n_ancillas)]
            post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)

            post_processed_patch = 2 * post_measurement_probs - 1
            return post_processed_patch


if __name__ == "__main__":
    # 定义生成器的参数
    image_size = 28
    channels = 1
    n_generators = 4
    n_gen_qubits = 10
    n_ancillas = 1
    n_gen_layers = 10
    patch_shape = (14, 14)

    # 实例化生成器
    quantum_generator = CAQ_QC(
        image_size=image_size,
        channels=channels,
        n_generators=n_generators,
        n_gen_qubits=n_gen_qubits,
        n_ancillas=n_ancillas,
        n_gen_layers=n_gen_layers,
        patch_shape=patch_shape,
        n_critic_qubits=5,
        n_critic_layers=1,
    ).generator

    noise = torch.randn(10, 10)
    generated_images = quantum_generator(noise)
    print(generated_images.shape)

