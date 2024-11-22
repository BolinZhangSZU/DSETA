import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Regressor, FeatureExtractor, Classifier
from attn import PositionalEncoding


class DSETA(nn.Module):
    def __init__(self, configs, config_link, config_trans, dropout=0.1):
        super().__init__()
        self.configs = configs
        self.config_link = config_link
        self.config_trans = config_trans

        d_model = config_trans["d_model"]
        for config in configs:
            setattr(self, f'fe_{config["name"]}', FeatureExtractor(config))
            feature_dim = getattr(self, f'fe_{config["name"]}').feature_dim
            setattr(self, f'linear_{config["name"]}', nn.Linear(feature_dim, d_model))

        self.fe_link = FeatureExtractor(config_link)
        self.linear_link = nn.Linear(self.fe_link.feature_dim, d_model)

        self.pe = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(**config_trans, dropout=dropout, batch_first=True)

        self.regressor = Regressor(d_model, dropout=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def load_ds_embedding(self, ds_para):
        self.fe_driver.embedding_DriverID.load_state_dict(torch.load(ds_para))

    def forward(self, dense, sparse, dense_link, sparse_link):
        outs = self.forward_transformer(dense, sparse, dense_link, sparse_link)

        outs_sum = torch.sum(outs, dim=1)

        out = self.regressor(outs_sum)
        return out  # (N, 1)

    def forward_transformer(self, dense, sparse, dense_link, sparse_link):
        features = []
        for config in self.configs:
            feature = getattr(self, f'fe_{config["name"]}')(dense, sparse)
            feature = F.relu(getattr(self, f'linear_{config["name"]}')(feature))
            features.append(feature.unsqueeze(1))
        src = torch.cat(features, 1)
        feature_link = self.fe_link(dense_link, sparse_link)
        tgt = self.pe(F.relu(self.linear_link(feature_link)))
        outs = self.transformer(src, tgt)
        return outs

    @staticmethod
    def generate_config():
        configs = [
            {
                "name": "space",
                "dense": {  # Distance:1, OGridEmb:32, DGridEmb:32
                    "size": 65,
                    "index": (0, 65)
                },
                "sparse": [
                    {"col": 0, "name": "OGridID", "size": 3782, "dim": 32},
                    {"col": 1, "name": "DGridID", "size": 3782, "dim": 32},
                ],
            },
            {
                "name": "time",
                "dense": {  # TimeEmb:32
                    "size": 32,
                    "index": (65, 65 + 32)
                },
                "sparse": [
                    {"col": 2, "name": "WeekID", "size": 7, "dim": 32},
                    {"col": 3, "name": "DayID", "size": 288, "dim": 32},
                ],
            },
            {
                "name": "driver",
                "dense": None,
                "sparse": [
                    {"col": 4, "name": "DriverID", "size": 1000, "dim": 32},
                ],
            },
        ]

        config_link = {
            "dense": {  # LinkDistance:1, LinkEmb:32
                "size": 33,
                "index": (0, 33)
            },
            "sparse": [
                {"col": 0, "name": "LinkID", "size": 43897, "dim": 32},
                {"col": 1, "name": "LinkRoadType", "size": 6, "dim": 32},
            ],
        }
        config_trans = {
            "d_model": 256,
            "nhead": 1,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 256
        }
        return configs, config_link, config_trans

    @staticmethod
    def generate_config_DS():
        configs = [
            {
                "name": "space",
                "dense": {  # Distance:1, OGridEmb:32, DGridEmb:32
                    "size": 65,
                    "index": (0, 65)
                },
                "sparse": [
                    {"col": 0, "name": "OGridID", "size": 3782, "dim": 32},
                    {"col": 1, "name": "DGridID", "size": 3782, "dim": 32},
                ],
            },
            {
                "name": "time",
                "dense": {  # TimeEmb:32
                    "size": 32,
                    "index": (65, 65 + 32)
                },
                "sparse": [
                    {"col": 2, "name": "WeekID", "size": 7, "dim": 32},
                    {"col": 3, "name": "DayID", "size": 288, "dim": 32},
                ],
            },
            {
                "name": "driver",
                "dense": {  # DriverEmb:32
                    "size": 32,
                    "index": (97, 97 + 32)
                },
                "sparse": [
                    {"col": 4, "name": "DriverID", "size": 1000, "dim": 32},
                ],
            },
        ]

        config_link = {
            "dense": {  # LinkDistances 1, link_emb 32
                "size": 33,
                "index": (0, 33)
            },
            "sparse": [
                {"col": 0, "name": "LinkID", "size": 43897, "dim": 32},
                {"col": 1, "name": "LinkRoadType", "size": 6, "dim": 32},
            ],
        }
        config_trans = {
            "d_model": 256,
            "nhead": 1,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 256
        }
        return configs, config_link, config_trans

    @staticmethod
    def generate_config_DSR():
        configs = [
            {
                "name": "space",
                "dense": {  # Distance:1, OGridEmb:32, DGridEmb:32
                    "size": 65,
                    "index": (0, 65)
                },
                "sparse": [
                    {"col": 0, "name": "OGridID", "size": 3782, "dim": 32},
                    {"col": 1, "name": "DGridID", "size": 3782, "dim": 32},
                ],
            },
            {
                "name": "time",
                "dense": {  # TimeEmb:32
                    "size": 32,
                    "index": (65, 65 + 32)
                },
                "sparse": [
                    {"col": 2, "name": "WeekID", "size": 7, "dim": 32},
                    {"col": 3, "name": "DayID", "size": 288, "dim": 32},
                ],
            },
            {
                "name": "driver",
                "dense": {  # DriverEmb:32
                    "size": 32,
                    "index": (97, 97 + 32)
                },
                "sparse": None
            },
        ]

        config_link = {
            "dense": {  # LinkDistances 1, link_emb 32
                "size": 33,
                "index": (0, 33)
            },
            "sparse": [
                {"col": 0, "name": "LinkID", "size": 43897, "dim": 32},
                {"col": 1, "name": "LinkRoadType", "size": 6, "dim": 32},
            ],
        }
        config_trans = {
            "d_model": 256,
            "nhead": 1,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 256
        }
        return configs, config_link, config_trans


class DSETA_ALL(DSETA):
    def __init__(self, configs, config_link, config_trans, c_rou, c_seg, dropout=0.1):
        super().__init__(configs, config_link, config_trans, dropout=dropout)
        d_model = config_trans["d_model"]
        self.classifier_rou = Classifier(d_model, c_rou, dropout=dropout)
        self.classifier_seg = Classifier(d_model, c_seg, dropout=dropout)

        dd = 120 // c_rou
        B = [i * dd for i in range(c_rou + 1)]
        self.b = torch.tensor([(B[i] + B[i + 1]) / 2 for i in range(c_rou)])

        self._reset_parameters()

    def dspr(self, y_):
        self.b = self.b.to(y_.device)
        mu = y_ @ torch.log(self.b)
        sigema_sqare = torch.sum(y_ * torch.pow(torch.log(self.b) - mu.unsqueeze(-1), 2), dim=-1)
        y_exp = torch.exp(mu + sigema_sqare / 2).unsqueeze(-1)
        return y_exp

    def forward(self, dense, sparse, dense_link, sparse_link):
        outs = self.forward_transformer(dense, sparse, dense_link, sparse_link)

        leng = outs.shape[1]
        cum = torch.tril(torch.ones((leng, leng), device=outs.device))
        outs_cum = cum @ outs

        out_rou = self.regressor(outs_cum)

        out_seg = self.regressor(outs)
        out_seg_sum = torch.sum(out_seg, dim=1)

        outs_sum = outs_cum[:, -1, :]
        out_ds_rou = self.classifier_rou(outs_sum)

        out_dspr = self.dspr(out_ds_rou)

        out_ds_seg = self.classifier_seg(outs).permute(0, 2, 1)

        #      (N, 1),  (N, L, 1), (N, 1),    (N, C),     (N, 1),   (N, C, L)
        return out_rou, out_seg, out_seg_sum, out_ds_rou, out_dspr, out_ds_seg


if __name__ == '__main__':
    from common.utils import count_parameters

    model = DSETA(*DSETA.generate_config())
    count_parameters(model)
