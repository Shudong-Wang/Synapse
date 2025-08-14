import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq, d_model)
        B, S, D = x.shape
        H = self.nhead
        Hd = self.head_dim

        # Project queries, keys, values
        q = self.q_proj(x).reshape(B, S, H, Hd).transpose(1, 2)  # (B, H, S, Hd)
        k = self.k_proj(x).reshape(B, S, H, Hd).transpose(1, 2)  # (B, H, S, Hd)
        v = self.v_proj(x).reshape(B, S, H, Hd).transpose(1, 2)  # (B, H, S, Hd)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)  # (B, H, S, S)
        if mask is not None:
            # mask: (B, S), True for padding positions, so we avoid them in attention
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))  # (B, H, S, S)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)  # (B, H, S, Hd)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)  # (B, S, D)

        output = self.out_proj(attn_output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        # src: (B, S, D), mask: (B, S)
        if self.norm_first:
            src2 = self.norm1(src)
            attn_out = self.self_attn(src2, mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            src2 = self.norm2(src)
            ff_out = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
            src = src + self.dropout3(ff_out)
        else:
            attn_out = self.self_attn(src, mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            src = self.norm1(src)
            ff_out = self.linear2(self.dropout2(self.activation(self.linear1(src))))
            src = src + self.dropout3(ff_out)
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        return output
