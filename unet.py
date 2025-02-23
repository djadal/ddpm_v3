from functools import partial
import numpy as np

from torch import nn
from torch.nn import Module, ModuleList
import torch.fft

from utils import default

from unet_blocks import *


# unet_model

class Unet1D(Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            dropout=0.,
            self_condition=False,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=4
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        # input_channels = channels * (2 if self_condition else 1)
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)
        self.cond_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:], [1024, 512, 256, 128]))  # [(init_dim, init_dim), (init_dim, init_dim*2), ...]

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.ref_mlp = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024)
        )
        
        # fft_block
        
        self.fft_block = nn.Sequential(
            nn.LayerNorm(3),
            nn.Conv1d(3, init_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(init_dim, init_dim, 3, padding=1),
        )

        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout) 

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out, seq) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                FFTBlock(dim_in, dim_in, patch_size=seq/4),
                Residual(PreNorm(dim_in, ReferenceModulatedCrossAttention(dim=seq))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                # downsample -> (b, 2*c, l/2)
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.target_dim = 12
        self.fusion_layer=nn.Linear(192, 1)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(dim=mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)
        self.embed_layer = nn.Embedding(
            num_embeddings=12, embedding_dim=64
        )  

        for ind, (dim_in, dim_out, seq) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                FFTBlock(dim_out, dim_out, patch_size=seq/4),
                Residual(PreNorm(dim_out, ReferenceModulatedCrossAttention(dim=seq))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)


    def time_embedding(self, d_model=128, device='cpu'):
        pe = torch.zeros(16, 1024, d_model).to(device) 

        pos = np.ones((16, 1024))
        pos = torch.tensor(pos, dtype=torch.float32).to(device)

        position = pos.unsqueeze(2) 
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)  
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, device):
        B = 16
        K = 12
        L = 1024

        time_embed = self.time_embedding(device=device)  # (B, L, time_emb)

        # feature_embed (B, K, feature_emb)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(device)
        )  # (K, feature_emb)
        feature_embed = feature_embed.unsqueeze(0).expand(B, -1, -1)  # (B, K, feature_emb)

        # (B, K, L, time_emb)
        time_embed = time_embed.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, L, time_emb)

        # (B, K, L, feature_emb)
        feature_embed = feature_embed.unsqueeze(2).expand(-1, -1, L, -1)  # (B, K, L, feature_emb)

 
        fused_info = torch.cat((feature_embed, time_embed), dim=-1)   # (B, K, L, time_emb)

        # fused_info = time_embed * feature_embed  # (B, K, L, time_emb)

        if fused_info.size(-1) != 1:
            fused_info = self.fusion_layer(fused_info)  # (B, K, L, 1)
            fused_info = fused_info.squeeze(-1)  # (B, K, L)

        return fused_info

    def forward(self, x, time, x_self_cond=None, reference=None):
        # x -> (b, c, l)
        if self.self_condition:
            # x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)  # x -> (b, 2*c, l)
     
        x = self.init_conv(x)  # x ->(b, init_dim, l)
        r = x.clone()  # r -> (b, init_dim, l)

        reference = self.ref_mlp(reference)
        reference = self.init_conv(reference)
        t = self.time_mlp(time)  # t ->(b, 4*init_dim)

        side_info = self.get_side_info(x.device)
        side_info = self.cond_conv(side_info)

        h = []
        ref = []
        side = []

        for block1, block2, attn, fft, downsample in self.downs:
            x = block1(x, t)
            reference = block1(reference, t)
            side_info = block1(side_info)

            h.append(x)
            side.append(side_info)
            ref.append(reference)

            x = block2(x, t)
            reference = block2(reference, t)
            side_info = block2(side_info, t)

            x = fft(x)

            x = attn(x, reference=reference, cond_info=side_info)

            h.append(x)
            ref.append(reference)
            side.append(side_info)

            x = downsample(x)
            reference = downsample(reference)
            side_info = downsample(side_info)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, fft, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            reference = torch.cat((reference, ref.pop()), dim=1)
            side_info = torch.cat((side_info, side.pop()), dim=1)

            x = block1(x, t)
            reference = block1(reference, t)
            side_info = block1(side_info, t)

            x = torch.cat((x, h.pop()), dim=1)
            reference = torch.cat((reference, ref.pop()), dim=1)
            side_info = torch.cat((side_info, side.pop()), dim=1)

            x = block2(x, t)
            reference = block2(reference, t)
            side_info = block2(side_info, t)

            x = fft(x)

            x = attn(x, reference=reference, cond_info=side_info)
            
            x = upsample(x)
            reference = upsample(reference)
            side_info = upsample(side_info) 


        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)