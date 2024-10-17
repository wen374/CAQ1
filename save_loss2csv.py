import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


disc_loss_0 = np.load('output/mosaiq/disc_loss_0.npy')
disc_loss_5 = np.load('output/mosaiq/disc_loss_5.npy')


# %% save to fig
plt.ion()
plt.plot(disc_loss_0)
plt.title('Disc Loss 0')
plt.show()
plt.savefig('output/mosaiq/disc_loss_0.png')
plt.close()

plt.ion()
plt.plot(disc_loss_5)
plt.title('Disc Loss 5')
plt.show()
plt.savefig('output/mosaiq/disc_loss_5.png')
plt.close()

# %% save to csv
pd.DataFrame({'disc_loss_0': disc_loss_0}).to_csv('output/mosaiq/disc_loss_0.csv', index=False)
pd.DataFrame({'disc_loss_5': disc_loss_0}).to_csv('output/mosaiq/disc_loss_5.csv', index=False)
