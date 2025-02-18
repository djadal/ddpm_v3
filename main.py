import argparse
from trainer import *

from unet import Unet1D
from diffusion_model import GaussianDiffusion1D
from datautils import Dataset_ECG_VIT

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="DDPM1d")
    
    # Trainer
    parser.add_argument("--data_path", type=str, default="E:\ECG\CPSC2018\Dataset\PTB_XL")
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--sample_interval", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--result_path", type=str, default='.\results')
    
    # U-Net
    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=9)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--in_c", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--condition", type=bool, default=True)
    parser.add_argument("--is_random_pe", type=bool, default=False)
    
    # GaussianDiffusion
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--objective", type=str, default="pred_noise", choices=["pred_noise", "pred_x0", "pred_v"])
    parser.add_argument("--beta", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--normalize", type=bool, default=False)
    
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
                                beta_schedule=args.beta,
                                auto_normalize=args.normalize
                                )
    
    train_set = Dataset_ECG_VIT(root_path=args.data_path, flag='train', seq_length=args.length, ref_path='E:\ECG\denoising-diffusion-pytorch-main\ddpm_v2\database')
    
    trainer = Trainer1D(diffusion_model=model, 
                        dataset=train_set, 
                        train_batch_size=args.batch_size, 
                        train_lr=args.lr,
                        save_and_sample_every=args.sample_interval
                        )
    
    trainer.train()