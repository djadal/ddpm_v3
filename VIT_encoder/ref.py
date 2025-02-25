import argparse
import os

import torch
import yaml

import models.encoder as encoder

from datautils import Dataset_ECG_VIT
from torch.utils.data import DataLoader


def load_config(args) -> dict:
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    for k, v in vars(args).items():
        if v and k in config:
            config[k] = v

    return config

def embedding(model, data_loader, device, config):
    embedding_list = []
    
    with torch.no_grad():
        for (samples, _) in data_loader:
            samples = samples.type(torch.FloatTensor)
            samples = samples.to(device, non_blocking=True)
            
            embed = model(samples)
            embedding_list.append(embed.cpu())
        
        embedding_list = torch.cat(embedding_list, dim=0)
        torch.save(embedding_list.float(), os.path.join(config['embedding_path'], f'{config["dataset"]}_embedding.pt'))

def create_ref(model, data_loader, database, ref_num, device, config):
    num = ref_num
    embedding = torch.load(os.path.join(config['embedding_path'], f'{config["dataset"]}_embedding.pt'))
    references = []
    
    with torch.no_grad():
        for (samples, _) in data_loader:
            samples = samples.type(torch.FloatTensor)
            samples = samples.to(device, non_blocking=True)
            
            sample_embed = model(samples)
            distances = torch.norm(sample_embed.cpu() - embedding, dim=1)
            _, idx = torch.topk(-1 * distances, num + 1)
            
            if config['flag'] == 'train':
                references.append(idx[1:])
            else:
                references.append(torch.cat([database.all[i, :, :1024] for i in idx[:ref_num]], dim=1))
        
        references = torch.cat([t.unsqueeze(0) for t in references], dim=0)
        os.makedirs(config['ref_path'], exist_ok=True)
        torch.save(references, os.path.join(config['ref_path'], f'{config["dataset"]}_{config["flag"]}_ref.pt'))


def process_ref(config, model, database, ref_num):
    if config['flag'] == 'train':
        dataset = database
    else:    
        dataset = Dataset_ECG_VIT(root_path=config['data_path'], flag=config['flag'], seq_length=2500)
        
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=(config['flag'] == 'train'))
    create_ref(model, data_loader, database, ref_num, config['device'], config)

def main(config):
    device = torch.device(config['device'])
    config['data_path'] = os.path.join(config['root_path'], config['dataset'])
    
    train_set = Dataset_ECG_VIT(root_path=config['data_path'], flag='train', seq_length=2500)
    
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] != "scratch":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
                
    model.to(device)
    model.eval()
    
    if config['task'] == 'embedding':
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        embedding(model, train_loader, config['device'], config)
    else:
        process_ref(config, model, train_set, ref_num=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Reference")
    
    parser.add_argument('--task', type=str, default='ref', choices=['embedding', 'ref'])
    parser.add_argument('--flag', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--dataset', type=str, default='PTB_XL', choices=['PTB_XL', 'CPSC'])
    parser.add_argument('--config_path',
                        default="./configs/downstream/st_mem.yaml",
                        type=str,
                        metavar='FILE',
                        help='YAML config file path')
    parser.add_argument('--root_path',
                        default="/root/autodl-tmp/DDPM/Dataset",
                        type=str,
                        metavar='PATH',
                        help='Path of val dataset')
    parser.add_argument('--encoder_path',
                        default="./st_mem/encoder.pth",
                        type=str,
                        metavar='PATH',
                        help='Pretrained encoder checkpoint')
    parser.add_argument('--embedding_path',
                        default="./st_mem",
                        type=str,
                        metavar='PATH',
                        help='Path to save embeddings')
    parser.add_argument('--ref_path',
                        default="./database",
                        type=str,
                        metavar='PATH',
                        help='Path to save references')
    
    args = parser.parse_args()
    
    main(config=load_config(args=args))