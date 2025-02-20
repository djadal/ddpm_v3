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

def embedding(model, data_loader, device, embedding_path):
    embedding_list = []
    
    with torch.no_grad():
        for (samples, _) in data_loader:
            samples = samples.type(torch.FloatTensor)
            samples = samples.to(device, non_blocking=True)
            
            embed = model(samples)
            # embed = rearrange(embed, 'b c n p -> b (c n p)')
            embedding_list.append(embed.cpu())
        
        embedding_list = torch.cat(embedding_list, dim=0)

        torch.save(embedding_list.float(), embedding_path)

def create_ref(model, data_loader, database, ref_num, device, embedding_path, ref_path):
    num = ref_num
    embedding = torch.load(embedding_path)
    references = []
    
    with torch.no_grad():
        for (samples, _) in data_loader:
            samples = samples.type(torch.FloatTensor)
            samples = samples.to(device, non_blocking=True)
            
            sample_embed = model(samples)
            # sample_embed = rearrange(sample_embed, 'b c n p -> b (c n p)')
            distances=torch.norm(sample_embed.cpu() - embedding,dim=1)
            _, idx=torch.topk(-1*distances, num + 1)
            
            # for i in idx[1]:
            references.append(database.predict[idx[1], :, :1024])
            # references.append(idx[1:])
        
        references = torch.cat([t.unsqueeze(0) for t in references], dim=0)
        os.makedirs(ref_path, exist_ok=True)
        torch.save(references, os.path.join(ref_path, 'ref.pt'))

def Reference(config):
    device = torch.device(config['device'])

    train_set = Dataset_ECG_VIT(root_path=config['data_path'], flag='train', seq_length=2500)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] != "scratch":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        # print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        # msg = model.load_state_dict(checkpoint_model, strict=False)

    # if config['mode'] == "linprobe":
    #     for _, p in model.named_parameters():
    #         p.requires_grad = False
    #     for _, p in model.head.named_parameters():
    #         p.requires_grad = True

    model.to(device)
    # print(f"Model = {model}")
    model.eval()
    if config['task'] == 'embedding':
        
        embedding(model, train_loader, config['device'], config['embedding_path'])
    else:
        test_dataset = Dataset_ECG_VIT(root_path=config['data_path'], flag='test', seq_length=2500)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        create_ref(model, test_loader, train_set, ref_num=3, device=config['device'],
               embedding_path=config['embedding_path'], ref_path=config['ref_path'])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Create Reference")
    
    parser.add_argument('--task', type=str, default='ref', choices=['embedding', 'ref'])
    parser.add_argument('--config_path',
                        default="./configs/downstream/st_mem.yaml",
                        type=str,
                        metavar='FILE',
                        help='YAML config file path')
    parser.add_argument('--data_path',
                        default="C:\\Dataset\\PTB_XL",
                        type=str,
                        metavar='PATH',
                        help='Path of val dataset')
    parser.add_argument('--encoder_path',
                        default="./st_mem/encoder.pth",
                        type=str,
                        metavar='PATH',
                        help='Pretrained encoder checkpoint')
    parser.add_argument('--embedding_path',
                        default="./st_mem/embedding.pt",
                        type=str,
                        metavar='PATH',
                        help='Path to save embeddings')
    parser.add_argument('--ref_path',
                        default="./VIT_encoder/database/evaluate",
                        type=str,
                        metavar='PATH',
                        help='Path to save references')
    
    args = parser.parse_args()
    
    Reference(config=load_config(args=args))