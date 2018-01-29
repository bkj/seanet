#!/usr/bin/env python

"""
    trainer.py
"""

from __future__ import print_function, division

import os
import sys
import json
import numpy as np
from time import time
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from seanet import SeaNet
from lr import LRSchedule
from helpers import progress_bar
import torch.multiprocessing as mp

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --
# IO

def make_dataloaders(root='data', train_size=1.0, train_batch_size=128, eval_batch_size=256, num_workers=8, seed=123123):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
    
    if train_size < 1:
        train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=train_size, random_state=seed)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=True,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
        )
        
        valloader = torch.utils.data.DataLoader(
            trainset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=True,
            shuffle=True,
        )
        
        valloader = None
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True,
        shuffle=False,
    )
    
    return {
        "train" : trainloader,
        "test" : testloader,
        "val" : valloader,
    }


# --
# Helpers

def train_epoch(model, opt, lr_scheduler, epoch, dataloader, gpu_id=0, verbose=True):
    _ = model.train()
    batches_per_epoch = len(dataloader)
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = Variable(data.cuda(gpu_id)), Variable(targets.cuda(gpu_id))
        
        # Set LR
        LRSchedule.set_lr(opt, lr_scheduler(epoch + batch_idx / batches_per_epoch))
        
        opt.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        opt.step()
        
        train_loss += loss.data[0]
        predicted = torch.max(outputs.data, 1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        if verbose:
            progress_bar(batch_idx, batches_per_epoch, 'Loss: %.3f | Acc: %.3f%%'
                % (train_loss/(batch_idx+1), 100.*correct/total))
    
    return float(correct) / total


def eval_epoch(model, dataloader, gpu_id=0, verbose=True):
    _ = model.eval()
    batches_per_epoch = len(dataloader)
    eval_loss, correct, total = 0, 0, 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = Variable(data.cuda(gpu_id), volatile=True), Variable(targets.cuda(gpu_id))
        
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        
        eval_loss += loss.data[0]
        predicted = torch.max(outputs.data, 1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        if verbose:
            progress_bar(batch_idx, batches_per_epoch, 'Loss: %.3f | Acc: %.3f%%'
                % (eval_loss/(batch_idx+1), 100.*correct/total))
    
    return float(correct) / total


def train(model, dataloaders, epochs=1, gpu_id=0, verbose=True, **kwargs):
    model = model.cuda(gpu_id)
    lr_scheduler = LRSchedule.sgdr(period_length=epochs, **kwargs)
    opt = torch.optim.SGD(model.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=5e-4)
    
    performance = []
    for epoch in range(epochs):
        train_acc = train_epoch(model, opt, lr_scheduler, epoch, dataloaders['train'], gpu_id=gpu_id, verbose=verbose)
        test_acc = eval_epoch(model, dataloaders['test'], gpu_id=gpu_id, verbose=verbose)
        if dataloaders['val']:
            val_acc = eval_epoch(model, dataloaders['val'], gpu_id=gpu_id, verbose=verbose)
        else:
            val_acc = None
        
        perf = {
            "epoch" : epoch,
            "train" : train_acc,
            "test"  : test_acc,
            "val"   : val_acc,
        }
        print(json.dumps(perf), file=sys.stderr)
        performance.append(perf)
        
        if train_acc < 0.2:
            print('*' * 50 + 'VV break vv ' + '*' * 50, file=sys.stderr)
            print(model, file=sys.stderr)
            print('*' * 50 + '^^ break ^^' + '*' * 50, file=sys.stderr)
            break
    
    return model.cpu(), performance


# --
# Run

def _mp_train_worker(run_name, step_id, models, model_ids, results, train_size, **kwargs):
    dataloaders = make_dataloaders(train_size=train_size)
    
    for model_id in model_ids:
        start_time = time()
        
        # --
        # Training
        
        model, performance = train(models[model_id], dataloaders, **kwargs)
        
        # --
        # Logging
        
        model_name = model.get_id() + datetime.now().strftime('-%Y%m%d_%H%M%S')
        model_path = os.path.join('results', run_name, 'models', model_name)
        log_path = os.path.join('results', run_name, 'logs', model_name)
        
        results[model_id] = {
            "timestamp"   : datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "run_name"    : str(run_name),
            "step_id"     : int(step_id),
            "model_id"    : str(model_id),
            "time"        : time() - start_time,
            "gpu_id"      : kwargs['gpu_id'],
            
            "performance" : performance,
            
            "model_path"  : str(model_path),
            "log_path"    : str(log_path),
        }
        print(results[model_id], file=sys.stderr)
        
        json.dump(results[model_id], open(log_path, 'w'))
        model.save(model_path)
        del model



def train_mp(models, num_gpus=2, **kwargs):
    
    manager = mp.Manager()
    results = manager.dict()
    
    processes = []
    for gpu_id, model_ids in enumerate(np.array_split(list(models.keys()), num_gpus)):
        kwargs.update({
            "models"    : models,
            "model_ids" : model_ids,
            "gpu_id"    : gpu_id,
            "results"   : results,
        })
        
        p = mp.Process(target=_mp_train_worker, kwargs=kwargs)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    results = dict(results)
    for k in models.keys():
        results[k]['model'] = SeaNet.load(results[k]['model_path'])
    
    return results