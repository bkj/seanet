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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from seanet import SeaNet
from lr import LRSchedule
from utils import progress_bar
import torch.multiprocessing as mp

# --
# IO

def make_dataloaders():
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=4,
        pin_memory=False
    )
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False,
    )
    
    return trainloader, testloader

# --
# Helpers

def train_epoch(model, opt, lr_scheduler, epoch, trainloader, gpu_id=0, verbose=True):
    _ = model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    batches_per_epoch = len(trainloader)
    
    for batch_idx, (data, targets) in enumerate(trainloader):
        data, targets = Variable(data.cuda(gpu_id)), Variable(targets.cuda(gpu_id))
        
        # Set LR
        LRSchedule.set_lr(opt, lr_scheduler(epoch + batch_idx / batches_per_epoch))
        
        opt.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        opt.step()
        
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        if verbose:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total


def test_epoch(model, testloader, gpu_id=0, verbose=True):
    _ = model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(testloader):
        data, targets = Variable(data.cuda(gpu_id), volatile=True), Variable(targets.cuda(gpu_id))
        
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        if verbose:
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total


def train(model, trainloader, testloader, epochs=1, gpu_id=0, verbose=True, **kwargs):
    if not kwargs.get('period_length', 0):
        kwargs['period_length'] = epochs
    
    model = model.cuda(gpu_id)
    lr_scheduler = LRSchedule.sgdr(**kwargs)
    opt = torch.optim.SGD(model.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=5e-4)
    
    train_accs, test_accs = [], []
    for epoch in range(0, epochs):
        print('epoch=%d' % epoch, file=sys.stderr)
        
        train_acc = train_epoch(model, opt, lr_scheduler, epoch, trainloader, gpu_id=gpu_id, verbose=verbose)
        test_acc = test_epoch(model, testloader, gpu_id=gpu_id, verbose=verbose)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    return model.cpu(), {"train_accs" : train_accs, "test_accs" : test_accs}


# --
# Run

def _mp_train_worker(run_name, models, model_ids, results, **kwargs):
    trainloader, testloader = make_dataloaders()
    for model_id in model_ids:
        t = time()
        
        # --
        # Training
        
        model, performance = train(models[model_id], trainloader, testloader, **kwargs)
        
        # --
        # Logging
        
        model_name = model.get_id() + datetime.now().strftime('-%Y%m%d_%H%M%S')
        model_path = os.path.join('results/models', model_name)
        log_path = os.path.join('results/logs', run_name, model_name)
        torch.save(model.state_dict(), model_path)
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        results[model_id] = {
            "run_name"    : run_name,
            "model"       : None,
            "model_path"  : model_path,
            "log_path"    : log_path,
            "performance" : performance,
            "timestamp"   : timestamp,
        }
        
        json.dump(results[model_id], open(log_path, 'w'))
        model.save(model_path)
        
        print({
            "run_name"          : run_name,
            "model_id"          : model_id,
            "time"              : time() - t,
            "performance_train" : performance['train_accs'][-1],
            # "performance_val"   : performance['val_accs'][-1],
            "performance_test"  : performance['test_accs'][-1],
            "gpu_id"            : kwargs['gpu_id'],
            "timestamp"         : timestamp,
        }, file=sys.stderr)
        
        del model



def train_mp(run_name, models, num_gpus=2, **kwargs):
    for d in ['results/models', 'results/logs']:
        if not os.path.exists(os.path.join(d, run_name)):
            os.makedirs(os.path.join(d, run_name))
    
    manager = mp.Manager()
    results = manager.dict()
    
    chunks = np.array_split(list(models.keys()), num_gpus)
    
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        kwargs.update({
            "run_name"  : run_name,
            "models"    : models,
            "model_ids" : chunk,
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