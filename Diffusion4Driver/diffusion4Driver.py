import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DRIVER_SIZE = 32
DRIVER_NUM = 1000
SEED = 0
set_seed(SEED)

BATCH_SIZE = 128
EPOCH_NUM = 1000
LR = 0.001

SAVE_MODEL = True
USE_WRITER = True

model_name = f"d4d_bs-{BATCH_SIZE}_lr-{LR}_sd-{SEED}"
print("Model:", model_name)
if not os.path.exists(f"./{model_name}"):
    os.mkdir(f"./{model_name}")

if SAVE_MODEL:
    model_path = f"./{model_name}/para"
    if not os.path.exists(model_path):
        os.mkdir(model_path)

if USE_WRITER:
    if not os.path.exists(f"./{model_name}/log"):
        os.mkdir(f"./{model_name}/log")
    lwriter = open(f"./{model_name}/log/{model_name}.log", mode='x')

# load dataset
# csv file:
# DriverID;distance;duration
# 0;0.031682801837145336;0.10828385637545943
# 0;0.018306709810003673;0.047780605032513426
min_s, max_s = 0.5000032340440607, 59.40272347663117
min_t, max_t = 60, 3597
data_df = pd.read_csv("./ddp.csv", sep=";", encoding="utf8")
dataset = torch.Tensor(data_df.values.tolist())

# Diffusion Hyper-parameters
num_steps = 100
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

# diffusion process
def q_x(x_0, t):
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alphas_l_m_t * noise


# denoise model
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


# loss function
def diffusion_loss_fn(model, x_0, d, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]

    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)

    a = alphas_bar_sqrt[t].to(DEVICE)
    am1 = one_minus_alphas_bar_sqrt[t].to(DEVICE)
    e = torch.randn_like(x_0).to(DEVICE)
    x = x_0.to(DEVICE) * a + e * am1
    output = model(x, t.squeeze(-1).to(DEVICE), d.to(DEVICE))
    return (e - output).square().mean()


# denoise process
def p_sample_loop(model, d, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]

    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, d, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, d, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    d = torch.ones(x.shape[0], dtype=torch.int) * d

    coeff = (betas[t] / one_minus_alphas_bar_sqrt[t]).to(DEVICE)
    eps_theta = model(x.to(DEVICE), t.to(DEVICE), d.to(DEVICE))
    mean = (1 / (1 - betas[t.cpu()].to(DEVICE)).sqrt()) * (x.to(DEVICE) - (coeff * eps_theta))
    z = torch.randn_like(x).to(DEVICE)
    sigma_t = betas[t.cpu()].sqrt().to(DEVICE)

    sample = mean + sigma_t * z
    return sample


# train
print("Training model...")

dataloder = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
plt.rc("text", color="blue")

model = MLPDiffusion(num_steps).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in tqdm(range(EPOCH_NUM)):
    for idx, batch_x in enumerate(dataloder):
        x_0 = batch_x[:, 1:]
        d = batch_x[:, 0].int()
        loss = diffusion_loss_fn(model, x_0, d, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    if SAVE_MODEL:
        path = os.path.join(model_path, f"{model_name}_ep-{epoch}.pkl")
        torch.save(model.state_dict(), path)

    print(f"Loss of epoch {epoch}: {loss.cpu().item()}")
    if USE_WRITER:
        lwriter.write(f"{epoch},{loss.cpu().item()}\n")

        drivers = [274, 892]
        colors = ['blue', 'red']
        x_seqs = []
        for d in drivers:
            sample_num = data_df.loc[data_df["DriverID"] == d, :].shape[0]
            x_seqs.append(
                p_sample_loop(model, d, torch.Size([sample_num, 2]), num_steps, betas, one_minus_alphas_bar_sqrt))

        fig, axs = plt.subplots(len(drivers), 10 + 1, figsize=(18, 2 * len(drivers)))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
        for j in range(len(drivers)):
            idx = data_df.loc[data_df["DriverID"] == drivers[j], :].index
            raw = dataset[idx[0]:idx[-1] + 1, 1:]
            axs[j, 10].scatter(raw[:, 1], raw[:, 0], color=colors[j], edgecolor="white", marker="o")
            s, t = [], []
            for k in range(raw.shape[0]):
                if 0 < raw[k, 0] < 1 and 0 < raw[k, 1] < 1:
                    s.append(raw[k, 0])
                    t.append(raw[k, 1])
            s, t = np.array(s), np.array(t)
            s = s * (max_s - min_s) + min_s
            t = (t * (max_t - min_t) + min_t) / 3600.
            v = s / t

            x, y = [0., 1.], [0., (max_t * v.mean() / 3600. - min_s) / (max_s - min_s)]
            axs[j, 10].plot(x, y, colors[j])

            v_mean_raw, v_min_raw, v_max_raw = v.mean(), v.min(), v.max()

            axs[j, 10].xaxis.set_visible(False)
            axs[j, 10].yaxis.set_visible(False)
            axs[j, 10].set_xlim(0., 1.)
            axs[j, 10].set_ylim(0., 1.)
            axs[j, 10].set_xticks([0., 0.5, 1.])
            axs[j, 10].set_yticks([0., 0.5, 1.])
            axs[j, 10].set_aspect('equal')
            if j == 0:
                axs[j, 10].set_title("GT")

            for i in range(1, 11):
                cur_x = x_seqs[j][i * 10].cpu().detach()
                axs[j, i - 1].scatter(cur_x[:, 1], cur_x[:, 0], color="black", edgecolor="white")
                if i == 10:
                    x, y = [0., 1.], [0., (max_t * v_mean_raw / 3600. - min_s) / (max_s - min_s)]
                    axs[j, i - 1].plot(x, y, colors[j])

                    s, t = [], []
                    for k in range(cur_x.shape[0]):
                        if 0 < cur_x[k, 0] < 1 and 0 < cur_x[k, 1] < 1:
                            s.append(cur_x[k, 0])
                            t.append(cur_x[k, 1])
                    s, t = np.array(s), np.array(t)
                    s = s * (max_s - min_s) + min_s
                    t = (t * (max_t - min_t) + min_t) / 3600.
                    v = s / t

                    x, y = [0., 1.], [0., (max_t * v.mean() / 3600. - min_s) / (max_s - min_s)]
                    axs[j, i - 1].plot(x, y, "yellow")

                if j != 0 or i != 1:
                    axs[j, i - 1].xaxis.set_visible(False)
                    axs[j, i - 1].yaxis.set_visible(False)
                axs[j, i - 1].set_xlim(0., 1.)
                axs[j, i - 1].set_ylim(0., 1.)
                axs[j, i - 1].set_xticks([0., 0.5, 1.])
                axs[j, i - 1].set_yticks([0., 0.5, 1.])
                axs[j, i - 1].set_aspect('equal')
                if j == 0:
                    axs[j, i - 1].set_title("$\mathbf{x}_{" + str(i * 10) + "}$")
        plt.savefig(f"./{model_name}/log/{epoch}.png")
        plt.close()

if USE_WRITER:
    lwriter.close()