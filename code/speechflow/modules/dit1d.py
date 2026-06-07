"""DiT-1D: Diffusion Transformer for 1D latent sequence generation with speaker conditioning."""

import torch
import torch.nn as nn
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class TokenSequenceEmbedder(nn.Module):
    def __init__(self, num_tokens, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(num_tokens, hidden_size)
        self.dropout_prob = dropout_prob
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, tokens):
        tokens = tokens.squeeze(-1)
        embeddings = self.embedding_table(tokens)
        if self.training and self.dropout_prob > 0:
            embeddings = self.drop(embeddings)
        return embeddings


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_1d_sincos_pos_embed(embed_dim, num_patches):
    pos = np.arange(num_patches, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class DiT1D(nn.Module):
    """1D Diffusion Transformer with speaker embedding conditioning (adaLN-Zero)."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        feat_dim=1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_tokens=4096,
        dropout_prob=0.1,
        learn_sigma=False,
        spk_dim=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.x_embedder = nn.Linear(in_channels * patch_size * feat_dim, hidden_size, bias=True)
        self.spk_embedder = nn.Linear(spk_dim, hidden_size, bias=True)
        self.num_patches = input_size // patch_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TokenSequenceEmbedder(num_tokens, hidden_size, dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels * feat_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.pos_embed.data.copy_(torch.from_numpy(
            get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        ).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.xavier_uniform_(self.spk_embedder.weight)
        nn.init.constant_(self.spk_embedder.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def patchify(self, x):
        B, C, T, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T // self.patch_size, self.patch_size * C * D)
        return x

    def unpatchify(self, x):
        B, T_p, _ = x.shape
        T = T_p * self.patch_size
        C, D = self.out_channels, self.feat_dim
        return x.reshape(B, T, C, D).permute(0, 2, 1, 3)

    def forward(self, x, t, y, spk):
        B = x.shape[0]
        x = self.x_embedder(self.patchify(x)) + self.pos_embed
        t_emb = self.t_embedder(t)
        if t_emb.shape[0] == 1 and B > 1:
            t_emb = t_emb.expand(B, -1)
        y_emb = self.y_embedder(y)[:, ::self.patch_size, :]
        spk_emb = self.spk_embedder(spk)
        c = t_emb.unsqueeze(1) + y_emb + spk_emb
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))


# Factory functions for standard model sizes
def DiT1D_XL_2(**kw): return DiT1D(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kw)
def DiT1D_B_2(**kw):  return DiT1D(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kw)
def DiT1D_S_2(**kw):  return DiT1D(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kw)

DiT1D_models = {
    'DiT1D-XL/2': DiT1D_XL_2,
    'DiT1D-B/2': DiT1D_B_2,
    'DiT1D-S/2': DiT1D_S_2,
}
