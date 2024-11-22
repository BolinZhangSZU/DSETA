import math
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if isinstance(self.config["sparse"], list) and \
                len(self.config["sparse"]) == 0:
            self.config["sparse"] = None

        if self.config["sparse"] is not None:
            for feature in self.config["sparse"]:
                setattr(self, f'embedding_{feature["name"]}',
                        nn.Embedding(feature["size"], feature["dim"]))

    @property
    def feature_dim(self) -> int:
        if self.config["dense"] is not None:
            dim = self.config["dense"]["size"]
        else:
            dim = 0
        if self.config["sparse"] is not None:
            for feature in self.config["sparse"]:
                dim += feature["dim"]
        return dim

    def forward(self, dense, sparse):
        # dense: (N, H) sparse: (N, H)
        if self.config["sparse"] is not None:
            sparse_features = []
            for feature in self.config["sparse"]:
                embed = getattr(self, f'embedding_{feature["name"]}')(
                    sparse[..., feature["col"]]
                )
                sparse_features.append(embed)
            sparse_feature = torch.cat(sparse_features, -1)
            if self.config["dense"] is not None:
                a = self.config["dense"]["index"][0]
                b = self.config["dense"]["index"][1]
                features = torch.cat([dense[..., a:b], sparse_feature], -1)
            else:
                features = sparse_feature
        else:
            if self.config["dense"] is not None:
                a = self.config["dense"]["index"][0]
                b = self.config["dense"]["index"][1]
                features = dense[..., a:b]
            else:
                raise Exception("No Features!")
        return features


class Regressor(nn.Module):
    def __init__(self, dim: int, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 4, dim // 8),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 8, 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Classifier(nn.Module):
    def __init__(self, dim: int, c: int, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 4, dim // 8),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 8, c),
            nn.Softmax(-1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
