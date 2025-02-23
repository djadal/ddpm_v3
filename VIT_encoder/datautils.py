import os
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

class Filter:
    def __init__(self,
                 filter_type: str = "lowpass",
                 cutoff: Optional[float] = None,
                 fs: Optional[float] = None,
                 order: int = 5) -> None:
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.filter = None

        if self.fs is None:
            raise ValueError("Sampling frequency must be specified.")

        if filter_type not in ["lowpass", "highpass", "bandpass"]:
            raise ValueError("filter_type must be 'lowpass', 'highpass', or 'bandpass'.")

        if filter_type in ["lowpass", "highpass"]:
            if not isinstance(cutoff, (float, int)):
                raise ValueError("cutoff must be a single frequency for 'lowpass' or 'highpass'.")
            self.filter = butter(order, cutoff, btype=filter_type, fs=fs, output="sos")

        elif filter_type == "bandpass":
            if not isinstance(cutoff, (tuple, list)) or len(cutoff) != 2:
                raise ValueError("cutoff must be a tuple of (low, high) frequencies for 'bandpass'.")
            self.filter = butter(order, cutoff, btype=filter_type, fs=fs, output="sos")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.filter is None:
            raise ValueError("Filter is not initialized properly.")
        else:
            return sosfiltfilt(self.filter, x, axis=-1)

class RandomCrop:
    """Crop randomly the input sequence.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, data) -> np.ndarray:
        x, y = data['data'], data['predict']
        if self.crop_length > x.shape[2]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[2]}).")
        start_idx = np.random.randint(0, x.shape[2] - self.crop_length + 1)
        
        return x[:, :, start_idx:start_idx + self.crop_length], y[:, :, start_idx:start_idx + self.crop_length]
    
class HeadCrop:
    """Keep head of the input sequence.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, data) -> np.ndarray:
        x, y = data['data'], data['predict']
        if self.crop_length > x.shape[2]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[2]}).")
        start_idx = 0
        
        return x[:, :, start_idx:start_idx + self.crop_length], y[:, :, start_idx:start_idx + self.crop_length]

def load_data_npy(file_path, crop):
    """
    :param file_path:
    :return: data_tensor, predict_tensor, indices_tensor
    """
    data = np.load(file_path)
    
    data_np, predict_np = crop(data)
    
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    predict_tensor = torch.tensor(predict_np, dtype=torch.float32)
    
    return data_tensor, predict_tensor   


class Dataset_ECG_VIT(Dataset):
    def __init__(self, root_path, flag, dataset= None, ref_path=None, seq_length=1024, random_crop=False):

        self.root_path = root_path
        self.ref_path = ref_path
        self.seq_length = seq_length
        if random_crop:
            self.crop = RandomCrop(seq_length)
        else:
            self.crop = HeadCrop(seq_length)
        
        self.flag = flag
        if flag == 'train':
            self.data_path = 'train_data.npz'
        elif flag == 'val':
            self.data_path = 'val_data.npz'
        elif flag == 'test':
            self.data_path = 'test_data.npz'
        else:
            raise ValueError('Flag error')
        
        self.seq, self.all = load_data_npy(os.path.join(self.root_path, self.data_path), crop=self.crop) # seq -> (I, II, V1)
        self.predict = self.all[:, [2, 3, 4, 5, 7, 8, 9, 10, 11], :] # (remove I, II, V1)
        
        if self.ref_path is not None:
            self.reference = torch.load(os.path.join(self.ref_path, f'{dataset}_{self.flag}_ref.pt'))
        else:
            self.reference = None    
            
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        if self.reference == None:
            return self.seq[idx], self.predict[idx]
        else:
            if self.flag == 'train':
                # ref_vec = []
                # for i in self.reference[idx]:
                #     ref_vec.append(self.predict[i])
                # ref_vec = torch.cat([ref.unsqueeze(0) for ref in ref_vec], dim=0) # -> (ref_num, 9, seq_length)
                # ref_vec = ref_vec[0]
                ref_vec = torch.cat([self.all[i] for i in self.reference[idx]], dim=1)
            else:
                ref_vec = self.reference[idx]

            # ref_vec = torch.mean(ref_vec, dim=0) # 对3条ref取平均
            
            return self.seq[idx], self.predict[idx], ref_vec 
