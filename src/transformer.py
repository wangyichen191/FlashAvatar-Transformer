import torch
import torch.nn as nn
from einops import rearrange as Rearrange
from einops.layers.torch import Rearrange as Rearrange_torch

class UVTransformer(nn.Module):
    def __init__(self, uv_size, patch_size, pixel_dim, dim, mlp_dim, \
                 out_dim, cond_dim, qk_dim, v_dim, depth=8, head=12, \
                 dropout=0., embed_dropout=0.):
        super().__init__()
        assert uv_size % patch_size == 0
        num_patches = (uv_size // patch_size) ** 2
        patch_dim = patch_size ** 2 * pixel_dim
        self.head = head

        self.to_patch_embedding = nn.Sequential(
            Rearrange_torch('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', \
                            p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.positional_embedding = nn.Parameter(
            torch.randn((1, num_patches, dim))
        )
        self.dropout = nn.Dropout(embed_dropout)
        self.transformer = Transformer(dim=dim, depth=depth, head=head, \
                                       mlp_dim=mlp_dim, cond_dim=cond_dim, \
                                       qk_dim=qk_dim, v_dim=v_dim, \
                                       dropout=dropout)
        self.mlp = nn.Linear(dim//head, out_dim)

    def forward(self, uvimage, cond):
        patch_embedding = self.to_patch_embedding(uvimage)
        patch_embedding += self.positional_embedding
        x = self.dropout(patch_embedding)
        x = self.transformer(x, cond)
        x = Rearrange(x, 'b n (h d) -> b (n h) d', h=self.head)
        return self.mlp(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, head, mlp_dim, cond_dim, \
                 qk_dim, v_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttention(dim=dim, cond_dim=cond_dim, qk_dim=qk_dim, \
                               v_dim=v_dim, head=head, dropout=dropout),
                FeedForward(dim=dim, mlp_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, cond):
        for (attention, feedforward) in self.layers:
            x = attention(x, cond) + x
            x = feedforward(x) + x
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, dim, cond_dim, qk_dim, v_dim, head, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_cond = nn.LayerNorm(cond_dim)
        
        self.proj_q = nn.Linear(cond_dim, qk_dim, bias=False)
        self.proj_k = nn.Linear(dim, head*qk_dim, bias=False)
        self.proj_v = nn.Linear(dim, head*v_dim, bias=False)
        self.head = head
        self.scale = qk_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(head*v_dim, dim),
            nn.Dropout(dropout)
        ) if not (head == 1 and dim == v_dim) else nn.Identity()

    def forward(self, x, cond):
        x = self.norm(x)
        cond = self.norm_cond(cond)
        q = self.proj_q(cond).repeat(1, x.shape[1], self.head)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = Rearrange(q, 'b n (h d) -> b h n d', h=self.head)
        k = Rearrange(k, 'b n (h d) -> b h n d', h=self.head)
        v = Rearrange(v, 'b n (h d) -> b h n d', h=self.head)

        w = torch.matmul(q, k.transpose(-2, -1))
        w *= self.scale
        w = self.softmax(w)

        o = torch.matmul(w, v)
        o = self.dropout(o)
        o = Rearrange(o, 'b h n d -> b n (h d)')
        return self.out(o)

        