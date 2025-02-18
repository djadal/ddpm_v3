from functools import partial
import numpy as np

from torch import nn
from torch.nn import Module, ModuleList

from utils import default

from unet_blocks import *

# unet_model

class Unet1D(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        # input_channels = channels * (2 if self_condition else 1)
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # [(init_dim, init_dim), (init_dim, init_dim*2), ...]

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout) # 预先定义好一个固定部分参数的类

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1) # downsample -> (b, 2*c, l/2)
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.target_dim = 128
        self.mid_attn = Residual(PreNorm(mid_dim, ReferenceModulatedCrossAttention(dim=mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)
        self.embed_layer = nn.Embedding(
            num_embeddings=128, embedding_dim=64
        )  # 将离散的目标转为连续的变量

        self.refconv = nn.Conv1d(
            in_channels=9,  # 输入序列长度
            out_channels=128,  # 输出序列长度
            kernel_size=1,  # 卷积核大小
            stride=1,  # 步长
            padding=0  # 填充
        )

        # 定义线性层，将特征维度从 1024 降为 128
        self.reflinear = nn.Linear(
            in_features=1024,  # 输入特征维度
            out_features=128  # 输出特征维度
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)


    def time_embedding(self, d_model=64, device='cpu'):
        pe = torch.zeros(16, 128, d_model).to(device)  # 形状（B,L,128)

        pos = np.ones((16, 128))
        pos = torch.tensor(pos, dtype=torch.float32).to(device)

        position = pos.unsqueeze(2)  # 【B，L，1】
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)  # 只给偶数维度地进行赋值
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe  # 【B，L，64】


    def get_side_info(self, device):
        B = 16
        K = 128
        L = 128  # B,321,126

        time_embed = self.time_embedding(device = device)  # (B,L,emb) 时间嵌入
        # time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)    #扩充到k个 (B,L,K,emb)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).expand(B, -1, -1)  # 把这个向量扩张成（B,K,64）

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,*)  （B.L.321,256)

        return side_info

    def forward(self, x, time, x_self_cond = None, reference = None):
        # x -> (b, c, l)
        if self.self_condition:
            # x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)# x -> (b, 2*c, l)

        x = self.init_conv(x) # x ->(b, init_dim, l)
        r = x.clone() # r -> (b, init_dim, l)

        t = self.time_mlp(time) # t ->(b, 4*init_dim)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        side_info = self.get_side_info(x.device)
        reference = self.refconv(reference)
        reference = self.reflinear(reference)

        x = self.mid_attn(x, reference=reference, cond_info=side_info)

        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)