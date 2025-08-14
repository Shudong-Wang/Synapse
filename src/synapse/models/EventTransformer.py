import torch
import torch.nn as nn

from .Transformer import TransformerEncoderLayer, TransformerEncoder

class EventTransformer(nn.Module):
    def __init__(self, num_particles=128, embed_dim=128, global_dim=10,
                 transformer_dim=32, num_heads=4, num_layers=2, dropout=0.05):
        super().__init__()
        # Combine x and v into a single per-particle feature vector
        self.input_proj = nn.Sequential(
            nn.Linear(5 + 4, embed_dim),  # (features + 4-momenta)
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learned position embedding
        self.position_embedding = nn.Embedding(num_particles, embed_dim)

        self.global_proj = nn.Sequential(
            nn.BatchNorm1d(global_dim),
            nn.Linear(global_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            # batch_first=True,  # Input: (batch, seq, feature)
            norm_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # # Norm for global features
        # self.global_norm = nn.BatchNorm1d(global_dim)

        # Final classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x, g, v, mask):
        # x: (N, 6, P)  v: (N, 4, P)  mask: (N, 1, P)
        # Permute to (N, P, *) for transformer (treat particles as sequence)
        N, _, P = x.shape
        x = x.permute(0, 2, 1)  # (N, P, 6)
        v = v.permute(0, 2, 1)  # (N, P, 4)
        inp = torch.cat([x, v], dim=-1)  # (N, P, 10)
        inp = self.input_proj(inp)  # (N, P, d)

        # Add position embedding
        pos_ids = torch.arange(P, device=x.device).unsqueeze(0).expand(N, P)  # (N, P)
        pos_emb = self.position_embedding(pos_ids)  # (N, P, embed_dim)
        inp = inp + pos_emb

        # Prepare mask for transformer: True for padding (so invert mask)
        # mask: (N, 1, P) -> (N, P)
        pad_mask = (mask.squeeze(1) == 0)  # (N, P)
        transformer_out = self.transformer(inp, src_key_padding_mask=pad_mask)  # (N, P, d)

        # Masked mean pooling over particles
        mask_float = mask.squeeze(1).float()  # (N, P)
        masked_sum = torch.sum(transformer_out * mask_float.unsqueeze(-1), dim=1)
        num_real = torch.sum(mask_float, dim=1, keepdim=True) + 1e-6
        pooled = masked_sum / num_real  # (N, d)

        # Normalize global features
        g = self.global_proj(g)

        # Concatenate pooled per-particle features and global features
        features = torch.cat([pooled, g], dim=1)  # (N, d + global_dim)

        # Classifier
        out = self.classifier(features)  # (N, 1)
        return out