import os, sys, shutil
import time
import json
import math
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset, DistributedSampler, Dataset


from lib import constants
from lib.model.transformer import MusicTransformer
from lib.inverse_power_with_warmup_scheduler import InversePowerWithWarmupLRScheduler
from lib.encoded_dataset import EncodedDataset
from lib.augmentations import MusicAugmentations

PAD_TOKEN = constants.TOKEN_PAD

params = dict(
    NAME = 'model_name',
    DS_FILE_PATH = 'ds_files.pt',
    SEED = 0,
    num_epochs = 100,
    batch_size = 2,
    num_workers = 0,
    val_every = 6000,
    save_every = 6000,
    lr = 1e-4,
    use_scheduler = True,
    peak_lr = 1e-4,
    warmup_steps = 4000,
    power = 2,
    shift = 100000,
    LOAD_NAME = '',
    LOG_TOTAL_NORM = True,
    CLIPPING = False,
    gpus = [0,1,2,3],
)

globals().update(params)


def create_dataloaders(batch_size, num_workers=0):
    '''Initializes augmentations, loads file lists to datasets and loaders and returns them'''
    print('loading data...')
    
    aug = MusicAugmentations()
    
    tr_dataset = YoutubeDataset(DS_FILE_PATH, transform=aug)
    vl_dataset = YoutubeDataset(DS_FILE_PATH, transform=None)
    np.random.seed(0)
    idxs = np.random.permutation(len(tr_dataset))
    vl, tr = np.split(idxs, [4000])
    train_dataset = Subset(tr_dataset, tr)
    val_dataset = Subset(vl_dataset, vl)
    
    sampler = DistributedSampler(train_dataset, world_size, rank, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=False, num_workers=num_workers)
    sampler = DistributedSampler(val_dataset, world_size, rank, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*4, sampler=sampler, pin_memory=False, num_workers=num_workers)
    
    return train_loader, val_loader
    
    
def init_model(lr, seed=0):
    '''Initializes model, loads weights if necessary and creates optimizer'''
    torch.manual_seed(seed)
    model = MusicTransformer(device, n_layers=12, d_model=1024, dim_feedforward=2048, num_heads=16, vocab_size=390+4, rpr=True).to(device)
    if LOAD_NAME != '':
        model.load_state_dict(torch.load(LOAD_NAME, map_location=device))
        print(f'Loaded model from {LOAD_NAME}')
    model = DistributedDataParallel(model, device_ids=[gpus[rank]])
    print(sum((torch.numel(x) for x in model.parameters()))/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    return model, optimizer

def validate(model, val_loader):
    CE = 0
    ACC = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, genre, idxs in val_loader:
            x[x==0] = PAD_TOKEN
            tgt = x.clone()
            x[:,-1] = constants.VOCAB_SIZE - 4 + genre
            x = torch.roll(x, 1, -1)
            x, tgt = x.to(device), tgt.to(device)

            logits = model(x)
            pred = logits.argmax(-1)

            mask = tgt != PAD_TOKEN
            n += mask.sum().item()
            CE += F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.flatten(), ignore_index=PAD_TOKEN, reduction='sum').item()
            ACC += (pred[mask] == tgt[mask]).sum().item()
            
    model.train()
    return CE/n, ACC/n

def train_ddp(rank_, world_size_):
    global device, NAME, SEED, rank, world_size
    rank, world_size = rank_, world_size_
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    
    device = torch.device(f'cuda:{gpus[rank]}')
    print(rank, gpus[rank], device)
    
    train_loader, val_loader = create_dataloaders(batch_size, num_workers)
    
    model, optimizer = init_model(lr, SEED)
    if use_scheduler:
        scheduler = InversePowerWithWarmupLRScheduler(optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps, power=power, shift=shift)
    
    if rank == 0:
        save_dir = f'output/{NAME}'
        save_name = f'{NAME}'
        if os.path.exists(save_dir):
            print(f'WARNING: {save_dir} exists! It may rewrite useful files')
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/{save_name}')
    
    # TRAIN
    LS = {'loss':[], 'lr':[], 'val_ce':[], 'val_acc':[]}

    i_val = 0
    i_step = -1
    best_ce = float('inf')
    patience = 0
    for ep in range(num_epochs):
        model.train()
        train_loader.sampler.set_epoch(ep)
        if rank == 0:
            bar = tqdm(train_loader, position=rank)
        else:
            bar = train_loader
        for x, genre, idxs in bar:
            i_step += 1
            x[x==0] = PAD_TOKEN
            tgt = x.clone()
            x[:,-1] = constants.VOCAB_SIZE - 4 + genre
            x = torch.roll(x, 1, -1)
            x, tgt = x.to(device), tgt.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.flatten(), ignore_index=PAD_TOKEN)
            
            optimizer.zero_grad()
            loss.backward()
            
            if CLIPPING:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING).item()
            else:
                total_norm = 0
            
            optimizer.step()
            
            if use_scheduler:
                scheduler.step()
                
            if i_step == warmup_steps - 1 and rank == 0:
                torch.save(model.module.state_dict(), f'{save_dir}/model_{save_name}_after_warmup.pt')

            if rank == 0:
                # logs
                LS['loss'] += [loss.item()]
                LS['lr'] += [optimizer.param_groups[0]['lr']]
                writer.add_scalar(f'Train/embedding_weight_norm', torch.norm(model.module.embedding.weight).item(), i_step)
                writer.add_scalar(f'Train/embedding_grad_norm', torch.norm(model.module.embedding.weight.grad).item(), i_step)
                writer.add_scalar(f'Train/output_weight_norm', torch.norm(model.module.Wout.weight).item(), i_step)
                writer.add_scalar(f'Train/output_grad_norm', torch.norm(model.module.Wout.weight.grad).item(), i_step)
                writer.add_scalar(f'Train/loss', loss.item(), i_step)
                writer.add_scalar(f'Train/perplexity', math.exp(loss.item()), i_step)
                writer.add_scalar(f'Train/lr', optimizer.param_groups[0]['lr'], i_step)
                if LOG_TOTAL_NORM:
                    total_norm = 0.
                    for p in model.module.parameters():
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    writer.add_scalar(f'Train/total_grad_norm', total_norm, i_step)
                bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'], norm=total_norm)
                

            # VALIDATION
            if i_step % val_every == val_every-1:
                val_ce, val_acc = validate(model, val_loader)
                if world_size > 1:
                    ce_all, acc_all = [[torch.zeros(1,device=device) for i in range(world_size)] for _ in range(2)]
                    [torch.distributed.all_gather(a, torch.tensor(x, dtype=torch.float32, device=device)) for a,x in zip([ce_all,acc_all], [val_ce,val_acc])]
                    val_ce, val_acc = [torch.cat(a).mean().item() for a in [ce_all,acc_all]]
                if rank == 0:
                    # log, save, patience tracking
                    LS['val_ce'] += [val_ce]
                    LS['val_acc'] += [val_acc]
                    writer.add_scalar(f'Val/ce', val_ce, i_val)
                    writer.add_scalar(f'Val/acc', val_acc, i_val)
                    writer.add_scalar(f'Val/perplexity', math.exp(val_ce), i_val)
                    if val_ce < best_ce:
                        patience = 0
                        best_ce = val_ce
                        torch.save({'history':LS,'epoch':ep,'params':params}, f'{save_dir}/hist_{save_name}_best.pt')
                        torch.save(model.module.state_dict(), f'{save_dir}/model_{save_name}_best.pt')
                    else:
                        patience += 1
                    print(f'{ep}: val_ce={val_ce}, val_acc={val_acc}, patience={patience}')
                i_val += 1

            # CHECKPOINT
            if (i_step % save_every == save_every-1) and rank == 0:
                torch.save({'history':LS,'epoch':ep,'params':params}, f'{save_dir}/hist_{save_name}.pt')
                torch.save(model.module.state_dict(), f'{save_dir}/model_{save_name}_{(i_step+1)//1000}k.pt')
    
    torch.distributed.destroy_process_group()
    

if __name__ == "__main__":
    print(NAME, SEED)
    world_size = len(gpus)
    torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
