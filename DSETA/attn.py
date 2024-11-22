import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (N, L, H)
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q:(N, L, H) K:(N, F, H) V:(N, F, H)
        d = Q.shape[-1]
        scores = Q @ K.transpose(1, 2) / math.sqrt(d)
        attn_weights = F.softmax(scores, dim=-1)  # ***
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ V
        return out, attn_weights


class Bert(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 activation=F.gelu,
                 dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                           nhead=num_attention_heads,
                                           dim_feedforward=intermediate_size,
                                           dropout=dropout,
                                           activation=activation,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_hidden_layers)

    def forward(self, x):
        return self.encoder(x)

    @staticmethod
    def get_config(bert_type="base"):
        bert_type_map = {
            "tiny": (128, 2, 128 // 64, 128 * 4),
            "mini": (256, 4, 256 // 64, 256 * 4),
            "small": (512, 4, 512 // 64, 512 * 4),
            "medium": (512, 8, 512 // 64, 512 * 4),
            "base": (768, 12, 768 // 64, 768 * 4),
            "large": (1024, 24, 1024 // 64, 1024 * 4),
        }
        return bert_type_map[bert_type.lower()]
