import os
import numpy as np

import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

def load_data_npy(file_path, seq_length):
    """
    :param file_path:
    :return: data_tensor, predict_tensor, indices_tensor
    """
    data = np.load(file_path)
    
    data_np = data['data'][:, :, :seq_length]
    predict_np = data['predict'][:, :, :seq_length]
    # indices_np = data['indices']
    
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    predict_tensor = torch.tensor(predict_np, dtype=torch.float32)
    
    return data_tensor, predict_tensor


def slice_sequence(data, seq_len, pred_len, shape):
    
    b, c, l = shape
    n = shape[2] - seq_len - pred_len + 1
    x = torch.zeros((shape[0], shape[1], n, seq_len), dtype=torch.float32)
    y = torch.zeros((shape[0], shape[1], n, pred_len), dtype=torch.float32)
    
    for i in range(n):
        x[:, :, i, :] = data[:, :, i:i+seq_len]
        y[:, :, i, :] = data[:, :, i+seq_len:i+seq_len+pred_len]
    
    x = x.transpose(1, 2)
    y = y.transpose(1, 2)
    x = x.reshape(b*n, c, seq_len)
    y = y.reshape(b*n, c, pred_len)    
    
    return x, y
    
class Dataset_ECG_VIT(Dataset):
    def __init__(self, root_path, flag, ref_path=None, seq_length=1024):

        self.root_path = root_path
        self.ref_path = ref_path
        self.seq_length = seq_length
        
        if flag == 'train':
            self.data_path = 'train_data.npz'
        elif flag == 'val':
            self.data_path = 'val_data.npz'
        elif flag == 'test':
            self.data_path = 'test_data.npz'
        else:
            raise ValueError('Flag error')
        
        self.seq, self.all = load_data_npy(os.path.join(self.root_path, self.data_path), self.seq_length) # seq -> (I, II, V1)
        self.predict = self.all[:, [2, 3, 4, 5, 7, 8, 9, 10, 11], :] # (remove I, II, V1)

        # self.seq, self.pred = slice_sequence(data, self.seq_len, self.pred_len, data.shape)
        
        if self.ref_path is not None:
            self.reference = torch.load(os.path.join(self.ref_path, 'ref.pt'))
        else:
            self.reference = None    
            
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        if self.reference == None:
            return self.seq[idx], self.predict[idx]
        else:
            ref_vec = []
            for i in self.reference[idx]:
                ref_vec.append(self.predict[i])
            ref_vec = torch.cat(ref_vec, dim=0) # -> (ref_num, 9, 2500)
            # ref_vec = ref_vec[:, [2, 3, 4, 5, 7, 8, 9, 10, 11], :]
            # return torch.cat([self.seq[idx], ref_vec], dim=1), self.all[idx][[0, 1, 6, 2, 3, 4, 5, 7, 8, 9, 10, 11], :]
            ref_vec = torch.mean(ref_vec, dim=0) # 对3条ref取平均
            return self.seq[idx], self.predict[idx], ref_vec # (16, 9, 1024)
