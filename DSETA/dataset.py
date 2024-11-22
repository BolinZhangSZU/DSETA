import os
import random
import torch
from torch.utils.data import Dataset


class TrajDatasetFromCache_ALL(Dataset):
    def __init__(self, path: str, leng: int):
        self.leng = leng
        self.C_R = 12
        self.C_S = 3

        self.label_t = torch.load(os.path.join(path, f"{self.leng}.label1.pth")).tolist()
        self.label_seg = torch.load(os.path.join(path, f"{self.leng}.label2.pth")).tolist()
        self.label_cum = torch.load(os.path.join(path, f"{self.leng}.label3.pth")).tolist()
        self.label_v = torch.load(os.path.join(path, f"{self.leng}.labelv.pth")).tolist()
        self.label_cr = torch.load(os.path.join(path, f"{self.leng}.labelds{self.C_R}.pth")).tolist()
        self.label_cs = torch.load(os.path.join(path, f"{self.leng}.labelc{self.C_S}.pth")).tolist()

        self.dense = torch.load(os.path.join(path, f"{self.leng}.dense.pth")).tolist()
        self.sparse = torch.load(os.path.join(path, f"{self.leng}.sparse.pth")).tolist()
        self.dense_link = torch.load(os.path.join(path, f"{self.leng}.dense_link.pth")).tolist()
        self.sparse_link = torch.load(os.path.join(path, f"{self.leng}.sparse_link.pth")).tolist()

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            label_t = torch.tensor([self.label_t[indexes]])  # (1, 1)
            label_seg = torch.tensor([self.label_seg[indexes]])  # (1, L, 1)
            label_cum = torch.tensor([self.label_cum[indexes]])  # (1, L, 1)
            label_v = torch.tensor([self.label_v[indexes]])  # (1, 1)
            label_cr = torch.tensor([self.label_cr[indexes]])  # (1)
            label_cs = torch.tensor([self.label_cs[indexes]])  # (1, L)

            dense = torch.tensor([self.dense[indexes]])  # (1, H)
            sparse = torch.tensor([self.sparse[indexes]])  # (1, H)
            dense_link = torch.tensor([self.dense_link[indexes]])  # (1, L, H)
            sparse_link = torch.tensor([self.sparse_link[indexes]])  # (1, L, H)
            return (dense, sparse, dense_link, sparse_link), \
                   (label_t, label_cum, label_seg, label_cr, label_v, label_cs)

        elif isinstance(indexes, (list, tuple)):
            label_t = torch.tensor([self.label_t[i] for i in indexes])  # (N, 1)
            label_seg = torch.tensor([self.label_seg[i] for i in indexes])  # (N, L, 1)
            label_cum = torch.tensor([self.label_cum[i] for i in indexes])  # (N, L, 1)
            label_v = torch.tensor([self.label_v[i] for i in indexes])  # (N, 1)
            label_cr = torch.tensor([self.label_cr[i] for i in indexes])  # (N)
            label_cs = torch.tensor([self.label_cs[i] for i in indexes])  # (N, L)

            dense = torch.tensor([self.dense[i] for i in indexes])  # (N, H)
            sparse = torch.tensor([self.sparse[i] for i in indexes])  # (N, H)
            dense_link = torch.tensor([self.dense_link[i] for i in indexes])  # (N, L, H)
            sparse_link = torch.tensor([self.sparse_link[i] for i in indexes])  # (N, L, H)

            return (dense, sparse, dense_link, sparse_link), \
                   (label_t, label_cum, label_seg, label_cr, label_v, label_cs)

    def __len__(self):
        return len(self.label_t)


def traj_dataloader(dataset, batch_size: int, shuffle: bool = True, bs: bool = False):
    num_sample = len(dataset)
    indexes = list(range(num_sample))
    if shuffle:
        random.shuffle(indexes)
    for i in range(0, num_sample, batch_size):
        batch_indexes = indexes[i: min(i + batch_size, num_sample)]
        if bs:
            yield dataset[batch_indexes], batch_indexes
        else:
            yield dataset[batch_indexes]


if __name__ == '__main__':
    pass
