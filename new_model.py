import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

# Feedforward and Attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return out

# Transformer

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# Numerical Embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# Main Class for Regression

class FTTransformer(nn.Module):
    def __init__(self, num_continuous, dim, depth, heads, dim_head = 16, dim_out = 1, attn_dropout = 0., ff_dropout = 0.):
        super().__init__()

        # Continuous inputs
        self.num_continuous = num_continuous
        self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # Output layer
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_numer):
        # Add numerically embedded tokens
        x_numer = self.numerical_embedder(x_numer)

        # Append CLS tokens
        b = x_numer.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x_numer), dim = 1)

        # Attend
        x = self.transformer(x)

        # Extract CLS token and compute output
        x = x[:, 0]
        logits = self.to_logits(x)

        return logits
