from pathlib import Path
from tqdm import tqdm
import random
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from unet_blocks import *
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from utils import num_to_groups, has_int_squareroot, cycle

import matplotlib.pyplot as plt


# trainer class

def plot_loss_curve(train_losses, val_losses=None, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='red', linestyle='-')
    plt.title('Training and Validation Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


class Trainer1D(object):
    def __init__(
            self,
            diffusion_model,
            dataset: Dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=16,
            results_folder='./results',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone, samples, target):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model_{milestone}.pt'))

        output = {
            'samples': samples,
            'target': target,
        }
        torch.save(output, str(self.results_folder / f'output_{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model_{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def evaluate(self, dataset, criterion, num_batches=None):
        accelerator = self.accelerator
        device = accelerator.device
        self.ema.ema_model.eval()

        eva_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        eva_loader = accelerator.prepare(eva_loader)

        eva_loader_list = list(eva_loader)

        if num_batches is None or num_batches > len(eva_loader_list):
            num_batches = len(eva_loader_list)

        selected_batches = random.sample(eva_loader_list, num_batches)

        with torch.no_grad():
            eva_loss = 0.

            for cond, target, ref in tqdm(selected_batches, desc="Evaluating", leave=False):
                cond, target, ref = cond.to(device), target.to(device), ref.to(device)
            
                sample = self.ema.ema_model.sample(batch_size=cond.shape[0], condition=cond, reference=ref)
                loss = criterion(sample, target)
                eva_loss += loss.item()

        eva_loss /= num_batches
    
        accelerator.print(f'evaluation loss: {eva_loss:.4f}')

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        loss_list = []

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    cond, target, ref = next(self.dl)
                    cond, target, ref = cond.to(device), target.to(device), ref.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(target, cond, ref)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                loss_list.append(total_loss)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and (self.step + 1) % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            cond, target, ref = next(self.dl)
                            cond, target, ref = cond.to(device), target.to(device), ref.to(device)
                            milestone = self.step // self.save_and_sample_every + 1
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(
                                map(lambda n: self.ema.ema_model.sample(batch_size=n, condition=cond, reference=ref), batches))

                        all_samples = torch.cat(all_samples_list, dim=0)

                        self.save(milestone, all_samples, target)

                pbar.update(1)

        plot_loss_curve(loss_list, save_path=str(self.results_folder / f'loss_curve.png'))
        accelerator.print('training complete')

