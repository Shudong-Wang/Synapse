import copy
import torch
import torch.nn as nn

from .ParticleTransformer import (
    SequenceTrimmer,
    Embed,
    GlobalEmbed,
    PairEmbed,
    BlockGlobalFeat,
    build_sparse_tensor,
    trunc_normal_,
)


class ParticleTransformerSimplePairwiseHeavy(nn.Module):
    """
    ParticleTransformer-like model with a stronger explicit pairwise path.

    Compared with ParticleTransformerSimple, this variant uses pairwise
    information in three places:
    1. attention bias, same as the original architecture;
    2. per-object pair context injected into token embeddings;
    3. per-event pair summary concatenated into the event classifier.
    """

    def __init__(
            self,
            input_dim,
            global_input_dim,
            num_classes=None,
            pair_input_dim=4,
            pair_extra_dim=0,
            remove_self_pair=False,
            use_pre_activation_pair=True,
            embed_dims=[128, 512, 128],
            global_embed_dims=[128, 512, 128],
            pair_embed_dims=[64, 64, 64],
            pair_context_dims=None,
            num_heads=8,
            num_layers=8,
            num_cls_layers=2,
            block_params=None,
            cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
            fc_params=[[64, 0], [256, 0], [64, 0]],
            activation='gelu',
            trim=True,
            for_inference=False,
            use_amp=False,
            **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.input_dim = input_dim
        self.global_input_dim = global_input_dim
        self.x_mean = None
        self.x_std = None
        self.g_mean = None
        self.g_std = None

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        assert global_embed_dims[-1] == embed_dims[-1] if len(global_embed_dims) > 0 else input_dim, \
            'The last dimension of global_embed_dims must match the last dimension of embed_dims.'

        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        if pair_context_dims is None:
            pair_context_dims = list(pair_embed_dims)

        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.global_embed = GlobalEmbed(global_input_dim, global_embed_dims, activation=activation) if len(global_embed_dims) > 0 else nn.Identity()

        # Original pairwise path: create attention bias per attention head.
        self.pair_embed = PairEmbed(
            pair_input_dim,
            pair_extra_dim,
            pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference,
        ) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None

        # New stronger pairwise path: produce dense pair context at embed_dim scale.
        self.pair_context_embed = PairEmbed(
            pair_input_dim,
            pair_extra_dim,
            pair_context_dims + [embed_dim],
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference,
        ) if pair_context_dims is not None and pair_input_dim + pair_extra_dim > 0 else None

        self.blocks = nn.ModuleList([BlockGlobalFeat(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([BlockGlobalFeat(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable scaling keeps the stronger pair path stable at initialization.
        self.pair_token_scale = nn.Parameter(torch.tensor(1.0))
        self.pair_event_scale = nn.Parameter(torch.tensor(1.0))
        self.pair_token_norm = nn.LayerNorm(embed_dim)
        self.pair_event_norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = 3 * embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        self.num_obj_classes = 2
        self.obj_fc = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, self.num_obj_classes)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    def setup_input_norm(self, x_mean, x_std, g_mean, g_std):
        self.x_mean = x_mean.view(1, -1, 1)
        self.x_std = x_std.view(1, -1, 1)
        self.g_mean = g_mean.view(1, -1)
        self.g_std = g_std.view(1, -1)

        assert self.x_mean.shape[1] == self.x_std.shape[1] == self.input_dim, \
            f'Input feature mean/std shape mismatch: {x_mean.shape[1]} vs {x_std.shape[1]} v.s {self.input_dim}'
        assert self.g_mean.shape[-1] == self.g_std.shape[-1] == self.global_input_dim, \
            f'Global feature mean/std shape mismatch: {g_mean.shape[-1]} vs {g_std.shape[-1]} v.s {self.global_input_dim}'

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    @staticmethod
    def _pair_context_summary(pair_context, mask):
        # pair_context: (N, C, P, P)
        # mask: (N, 1, P)
        pair_mask = mask.bool().unsqueeze(-1) & mask.bool().unsqueeze(-2)  # (N, 1, P, P)
        pair_weight = pair_mask.to(pair_context.dtype)

        pair_context = pair_context * pair_weight

        node_norm = pair_weight.sum(dim=-1).clamp(min=1.0)  # (N, 1, P)
        pair_token_context = pair_context.sum(dim=-1) / node_norm  # (N, C, P)

        event_norm = pair_weight.sum(dim=(-1, -2)).clamp(min=1.0)  # (N, 1)
        pair_event_context = pair_context.sum(dim=(-1, -2)) / event_norm  # (N, C)

        return pair_token_context, pair_event_context

    def forward(self, x, g, v=None, mask=None, uu=None, uu_idx=None):
        g = (g - self.g_mean.to(g.device)) / self.g_std.to(g.device)
        x = (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

        with torch.no_grad():
            if not self.for_inference and uu_idx is not None:
                uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            g = self.global_embed(g)

            attn_mask = None
            pair_token_context = None
            pair_event_context = None

            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))

            if (v is not None or uu is not None) and self.pair_context_embed is not None:
                pair_context = self.pair_context_embed(v, uu)  # (N, C, P, P)
                pair_token_context, pair_event_context = self._pair_context_summary(pair_context, mask)
                pair_token_context = self.pair_token_norm(pair_token_context.permute(2, 0, 1).contiguous())
                x = x + self.pair_token_scale * pair_token_context

            for block in self.blocks:
                x = block(x, x_cls=None, x_glob=None, padding_mask=padding_mask, attn_mask=attn_mask)

            cls_tokens = self.cls_token.expand(1, x.size(1), -1)
            g = g.expand(1, x.size(1), -1)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, x_glob=g, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)
            g = g.squeeze(0)

            if pair_event_context is None:
                pair_event_context = torch.zeros_like(x_cls)
            pair_event_context = self.pair_event_norm(pair_event_context)

            fused_cls = torch.cat((x_cls, g, self.pair_event_scale * pair_event_context), dim=1)

            if self.fc is None:
                return fused_cls

            output = self.fc(fused_cls)

            if pair_token_context is None:
                pair_token_context = torch.zeros_like(x)
            obj_repr = torch.cat((x, pair_token_context), dim=-1)
            obj_output = self.obj_fc(obj_repr.permute(1, 0, 2).contiguous().view(-1, obj_repr.size(-1)))
            output_for_aux = None

            return output, output_for_aux, obj_output


class ParticleTransformerPairwiseHeavy(ParticleTransformerSimplePairwiseHeavy):
    pass
