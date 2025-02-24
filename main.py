import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from trainer import Trainer1D

from unet import Unet1D
from diffusion_model import GaussianDiffusion1D
from VIT_encoder.datautils import Dataset_ECG_VIT
from utils import default


def plot_sample(args):
    output = torch.load('./results/samples/samples_{}_{}.pt'.format(args.timesteps, args.loss))

    lead_names = ["III", "aVR", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6"]

    samples = output['samples'].cpu()
    targets = output['target'].cpu()

    indices = torch.randperm(samples.shape[0])[:min(25, samples.shape[0])]

    for n, idx in enumerate(indices):
        sample, target = samples[idx], targets[idx]
        plt.figure(figsize=(32, 16))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(sample[i], label=f'Sample_{lead_names[i]}', color='red', linewidth=1)
            plt.plot(target[i], label=f'Target_{lead_names[i]}', color='blue', linewidth=1)
            plt.legend(loc="upper right")
            plt.ylabel("Amplitude(mV)")

        plt.xlabel("Time")

        plt.tight_layout()

        plt.savefig(f'/root/autodl-tmp/DDPM/Remote/figures/sample_{n}.png')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="DDPM1d")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default='PTB_XL', choices=["PTB_XL", "CPSC"])
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/DDPM/Dataset")
    parser.add_argument("--ref_path", type=str, default="./VIT_encoder/database")
    
    # Trainer
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--sample_interval", type=int, default=10000)
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
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--sampling_timesteps", type=int, default=50, help='Use DDIM Sampling when < timesteps')
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

    parser.add_argument("--status", type=str, default="train", choices=["train", "test", "sample"])
    parser.add_argument("--resume", type=int, default=0, help='Resume model from checkpoint')
    parser.add_argument("--loss", type=float, default=0., help='Resume model from checkpoint')
    parser.add_argument("--criterion", type=str, default='MSELoss', choices=criterion_dict.keys())

 
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
    
    train_set = Dataset_ECG_VIT(root_path=str(Path(args.data_path) / args.dataset), flag='train', 
                                seq_length=args.length, dataset=args.dataset, ref_path=args.ref_path)
    val_set = Dataset_ECG_VIT(root_path=str(Path(args.data_path) / args.dataset), flag='val', 
                              seq_length=args.length, dataset=args.dataset, ref_path=args.ref_path)
    test_set = Dataset_ECG_VIT(root_path=str(Path(args.data_path) / args.dataset), flag='test', 
                               seq_length=args.length, dataset=args.dataset, ref_path=args.ref_path)
    
    trainer = Trainer1D(diffusion_model=model, 
                        train_set=train_set,
                        val_set=val_set,
                        test_set=test_set, 
                        train_batch_size=args.batch_size,
                        train_num_steps=args.train_steps,
                        train_lr=args.lr,
                        criterion=criterion_dict[args.criterion],
                        save_and_sample_every=args.sample_interval
                        )
    
    if args.status == 'train':
        trainer.train()
    elif args.status == 'test':
        trainer.load(args.resume, args.sampling_timesteps, status=args.status)
        trainer.evaluate(trainer.val, criterion=trainer.criterion)
        trainer.evaluate(trainer.test, criterion=trainer.criterion)

    else: # sample
        trainer.load(args.resume, args.sampling_timesteps, status=args.status)
        trainer.sample(trainer.val, loss=args.loss)
        
        plot_sample(args)
        