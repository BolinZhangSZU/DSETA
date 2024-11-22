import numpy as np
import torch
import torch.nn as nn

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

DRIVER_SIZE = 32
DRIVER_NUM = 1000


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_dirver=DRIVER_NUM, num_units=DRIVER_SIZE):
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )
        self.step_embedding = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )
        self.driver_embedding = nn.ModuleList(
            [
                nn.Embedding(num_dirver, num_units),
            ]
        )

    def forward(self, x, t, d):
        for idx, embedding_layer in enumerate(self.step_embedding):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            if idx == 0:
                d_embedding = self.driver_embedding[idx](d)
                x += d_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)

        return x


model = MLPDiffusion(100)
model.load_state_dict(torch.load("./d4d.pkl"))

driver_embed = model.driver_embedding[0]
driver_embed.eval()
did = torch.tensor([i for i in range(1000)])
demb = driver_embed(did).detach().numpy()

# speed = np.zeros(1000, dtype=float)
# in_root_path = fr"F:\shanghai\traindata\d1000\driver"
# for driver_id in tqdm(range(1000)):
#     in_path = os.path.join(in_root_path, f"{driver_id}.csv")
#     trajs = pd.read_csv(in_path, sep=";", encoding="utf8")
#     speed[driver_id] = trajs["Speed"].values.mean()
# np.save("./speed.npy", speed)
speed = np.load("./speed.npy")
ind = np.argsort(speed)

y = np.zeros(1000, dtype=float)
for i in range(1000):
    y[ind[i]] = i

X = demb

tsne = TSNE(2)
X_tsne = tsne.fit_transform(X)

kmeans = KMeans(n_clusters=8)
kmeans.fit(X)
labels = kmeans.labels_

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

im1 = axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=20, cmap='RdBu_r', marker="o", alpha=0.8, vmax=1000)
im2 = axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels.astype(float), cmap="Set2", s=20, marker="o", alpha=0.8)

axs[0].xaxis.set_visible(False)
axs[0].yaxis.set_visible(False)
axs[1].xaxis.set_visible(False)
axs[1].yaxis.set_visible(False)
plt.colorbar(im1)
plt.colorbar(im2)
plt.show()
