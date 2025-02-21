import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from trainer import Trainer1D

from unet import Unet1D
from diffusion_model import GaussianDiffusion1D
from datautils import Dataset_ECG_VIT
from utils import default

from VIT_encoder.ref import load_Config, Reference

def plot_one_sample(args):
    # output = torch.load(str('./results/output_{}.pt'.format(int(args.train_steps / args.sample_interval))))
    output = torch.load('./results/evaluate.pt')
    os.makedirs('./results/figures/evaluation', exist_ok=True)

    lead_names = ["III", "aVR", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6"]

    for idx, (sample, target) in enumerate(zip(output['samples'].cpu(), output['target'].cpu())):
        plt.figure(figsize=(40, 16))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(sample[i], label=f'Sample_{lead_names[i]}', color='red', linewidth=1)
            plt.plot(target[i], label=f'Target_{lead_names[i]}', color='blue', linewidth=1)
            plt.legend(loc="upper right")
            plt.ylabel("Amplitude")

        plt.xlabel("Time Steps")
        plt.suptitle("Generated ECG Signals")

        plt.tight_layout()

        # plt.savefig(f'./results/figures/output_{idx}_{args.train_steps}.png')
        plt.savefig(f'./results/figures/evaluation/{idx}.png')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="DDPM1d")
    
    # Trainer
    parser.add_argument("--data_path", type=str, default="C:\\Dataset\\PTB_XL")
    parser.add_argument("--train_steps", type=int, default=20000)
    parser.add_argument("--sample_interval", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--result_path", type=str, default='.\results')
    
    # U-Net
    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=9)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--in_c", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--condition", type=bool, default=True)
    parser.add_argument("--is_random_pe", type=bool, default=False)
    
    # GaussianDiffusion
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=1000, help='Use DDIM Sampling when < timesteps')
    parser.add_argument("--ddim_eta", type=float, default=0., help='DDIM eta')
    parser.add_argument("--objective", type=str, default="pred_noise", choices=["pred_noise", "pred_x0", "pred_v"])
    parser.add_argument("--beta", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--normalize", type=bool, default=False)

    # Evaluate
    criterion_dict = {
    'MSELoss': nn.MSELoss(),
    'L1Loss': nn.L1Loss(),
    'CrossEntropyLoss': nn.CrossEntropyLoss(),
    }

    parser.add_argument("--resume", type=int, default=0, help='Resume model from checkpoint')
    parser.add_argument("--criterion", type=str, default='MSELoss', choices=criterion_dict.keys())
    parser.add_argument('--task', type=str, default='ref', choices=['embedding', 'ref'])
    parser.add_argument('--config_path',
                        default="./VIT_encoder/configs/downstream/st_mem.yaml",
                        type=str,
                        metavar='FILE',
                        help='YAML config file path')
    parser.add_argument('--encoder_path',
                        default="./VIT_encoder/st_mem/encoder.pth",
                        type=str,
                        metavar='PATH',
                        help='Pretrained encoder checkpoint')
    parser.add_argument('--embedding_path',
                        default="./VIT_encoder/st_mem/embedding.pt",
                        type=str,
                        metavar='PATH',
                        help='Path to save embeddings')
    parser.add_argument('--ref_path',
                        default="./VIT_encoder/database/evaluate",
                        type=str,
                        metavar='PATH',
                        help='Path to save references')
    
    args = parser.parse_args()
    
    unet = Unet1D(dim=args.dim, 
                  init_dim=args.init_dim, 
                  out_dim=args.out_dim,
                  channels=args.in_c,
                  dropout=args.dropout,
                  self_condition=args.condition,
                  random_fourier_features=args.is_random_pe
                  )
    
    model = GaussianDiffusion1D(model=unet,
                                seq_length=args.length,
                                timesteps=args.timesteps,
                                sampling_timesteps=args.sampling_timesteps,
                                beta_schedule=args.beta,
                                ddim_sampling_eta=args.ddim_eta,
                                auto_normalize=args.normalize
                                )
    
    train_set = Dataset_ECG_VIT(root_path=args.data_path, flag='train', seq_length=args.length, 
                                ref_path='.\database')
    
    trainer = Trainer1D(diffusion_model=model, 
                        train_set=train_set, 
                        train_batch_size=args.batch_size,
                        train_num_steps=args.train_steps,
                        train_lr=args.lr,
                        criterion=criterion_dict[args.criterion],
                        save_and_sample_every=args.sample_interval
                        )
    
    trainer.train()

    # trainer.load(args.resume, args.sampling_timesteps)
    trainer.evaluate(dataset=train_set, criterion = criterion_dict[args.criterion], num_batches=10)

    val_set = Dataset_ECG_VIT(root_path=args.data_path, flag='val', seq_length=args.length, 
                              ref_path='.\database\evaluation')
    trainer.evaluate(dataset=val_set, criterion = criterion_dict[args.criterion], num_batches=1)

    plot_one_sample(args)
    