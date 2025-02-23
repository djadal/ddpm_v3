import torch.nn as nn
import torch
from torch.nn import Module, ModuleList
from torch import einsum
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from utils import default, exists

from einops import rearrange,repeat
import math
# small helper modules

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class FFTBlock(Module):
    def __init__(self, dim_in=16, dim_out=None, patch_size=64,):
        super().__init__()
        self.to_patch = nn.Sequential(Rearrange('b c (n p) -> b c n p', p=patch_size),
                                      nn.LayerNorm(patch_size),
                                      Rearrange('b c n p -> (b n) c p')) 
        self.unpatch = Rearrange('(b n) c p -> b c (n p)', p=patch_size)
        self.fft = torch.fft.fft    

        # frequency_domain
        self.real_conv1 = nn.Conv1d(dim_in, 4*dim_in, 1) 
        self.img_conv1 = nn.Conv1d(dim_in, 4*dim_in, 1)
        self.real_conv2 = nn.Conv1d(4*dim_in, dim_out, 1) 
        self.img_conv2 = nn.Conv1d(4*dim_in, dim_out, 1)

        self.real_pool = nn.AdaptiveAvgPool1d(1)
        self.img_pool = nn.AdaptiveAvgPool1d(1)

        self.fre_conv = nn.Conv1d(dim_out, dim_out, 1)

        # time_domain
        self.time_conv = nn.Sequential(nn.Conv1d(dim_in, 4*dim_in, 1),
                                       nn.GELU(),
                                       nn.Conv1d(4*dim_in, dim_out, 1))

        self.final_conv = nn.Conv1d(dim_out, dim_out, 1)

        self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        x_f, x_t = torch.chunk(x, 2, dim=-1)

        # frequency domain
        # wfca block
        res_f = x_f
        x_f = self.to_patch(x_f)
        x_f = self.fft(x_f)
        real, img = x_f.real, x_f.imag

        real_1 = self.act(self.real_conv1(real) - self.img_conv1(img))
        img_1 = self.act(self.real_conv1(img) + self.img_conv1(real))
        
        real_2 = self.real_conv2(real_1) - self.img_conv2(img_1)
        img_2 = self.real_conv2(img_1) + self.img_conv2(real_1)

        real_attn, img_attn = self.real_pool(real_2), self.img_pool(img_2)

        x_f = torch.complex(real_2 * (1. + real_attn), img_2 * (1. + img_attn))
        x_f = torch.fft.ifft(x_f).real
        x_f = self.unpatch(x_f) + res_f

        # layernorm block
        x_f = self.fre_conv(nn.Layernorm(x_f.shape[2])(x_f)) + x_f

        # time domain
        x_t = self.time_conv(x_t) + x_t

        x = torch.cat((x_f, x_t), dim=-1)
        return self.final_conv(x) + residual


# def Upsample(dim, dim_out=None):
#     return nn.Sequential(
#         nn.Upsample(scale_factor = 2, mode = 'nearest'),
#         nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
#     )
    
def Upsample(dim, dim_out = None):
    return nn.ConvTranspose1d(dim, default(dim_out, dim), 4, stride=2, padding=1)

def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, stride=2, padding=1)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1)) 

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0. ,kernel_size=3, padding=1):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel_size=kernel_size, padding=padding) # same padding
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb) # (b, dim_out*2)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1) # ((b, dim_out, 1), (b, dim_out, 1))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class ReferenceModulatedCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            context_dim=None,
            dropout=0.,
            talking_heads=False,
            prenorm=False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.y_to_q = nn.Linear(dim, inner_dim, bias=False)
        self.cond_to_k = nn.Linear(2*dim + context_dim, inner_dim, bias=False)
        self.ref_to_v = nn.Linear(dim + context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(
            self,
            x,
            cond_info,
            reference,
            return_attn=False,
    ):
        # B, C, K, L, h, device = x.shape[0], x.shape[1], x.shape[2], x.shape[-1],
        h = self.heads

        x = self.norm(x)
        reference = self.norm(reference)
        cond_info = self.context_norm(cond_info)

        reference = repeat(reference, 'b n c -> (b f) n c', f=1)  # (B*C, K, L)
        q_y = self.y_to_q(x)  # (B*C,K,ND)

        cond = self.cond_to_k(torch.cat((x,cond_info, reference), dim=-1))  # (B*C,K,ND)
        ref = self.ref_to_v(torch.cat((x, reference), dim=-1))  # (B*C,K,ND)
        q_y, cond, ref = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q_y, cond, ref))  # (B*C, N, K, D)
        sim = einsum('b h i d, b h j d -> b h i j', cond, ref) * self.scale  # (B*C, N, K, K)
        attn = sim.softmax(dim=-1)
        context_attn = sim.softmax(dim=-2)
        # dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, ref)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, cond)
        # merge heads and combine out
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        if return_attn:
            return out, context_out, attn, context_attn

        return out


class LinearReferenceModulatedCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            context_dim=None,
            dropout=0.,
            talking_heads=False,
            prenorm=False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.y_to_q = nn.Linear(dim, inner_dim, bias=False)
        self.cond_to_k = nn.Linear(2 * dim + context_dim, inner_dim, bias=False)
        self.ref_to_v = nn.Linear(dim + context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(
            self,
            x,
            cond_info,
            reference,
            return_attn=False,
    ):
        h = self.heads

        # Normalize inputs
        x = self.norm(x)
        reference = self.norm(reference)
        cond_info = self.context_norm(cond_info)

        # Repeat reference for broadcasting
        reference = repeat(reference, 'b n c -> (b f) n c', f=1)

        # Project inputs to queries, keys, and values
        q_y = self.y_to_q(x)  # (B*C, K, ND)
        cond = self.cond_to_k(torch.cat((x, cond_info, reference), dim=-1))  # (B*C, K, ND)
        ref = self.ref_to_v(torch.cat((x, reference), dim=-1))  # (B*C, K, ND)

        # Rearrange for multi-head attention
        q_y, cond, ref = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q_y, cond, ref))

        # Linear Attention
        # 1. Compute kernelized queries and keys
        q_y = torch.nn.functional.elu(q_y) + 1  # Apply ELU activation for positive values
        cond = torch.nn.functional.elu(cond) + 1

        # 2. Compute attention scores using dot product
        sim = einsum('b h i d, b h j d -> b h i j', q_y, cond) * self.scale

        # 3. Normalize attention scores
        attn = sim.softmax(dim=-1)
        context_attn = sim.softmax(dim=-2)

        # Apply dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # Apply talking heads (if enabled)
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # Compute outputs
        out = einsum('b h i j, b h j d -> b h i d', attn, ref)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, cond)

        # Merge heads and project outputs
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out



