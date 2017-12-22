#!/usr/bin/env python

"""
    main.py
"""

from __future__ import print_function, division

import os
import sys
import json
import copy
import argparse
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
import morph_layers as mm
from lr import LRSchedule
from morph_ops import do_random_morph
from utils import progress_bar
from helpers import set_seeds, to_numpy, colstring

import torch.multiprocessing as mp

# torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default='./results/models/delete-me')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-init', type=float, default=0.05)
    
    return parser.parse_args()

args = parse_args()

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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
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
        print("gpu_id=%d | Epoch=%d" % (gpu_id, epoch), file=sys.stderr)
        
        train_acc = train_epoch(model, opt, lr_scheduler, epoch, trainloader, gpu_id=gpu_id, verbose=verbose)
        test_acc = test_epoch(model, testloader, gpu_id=gpu_id, verbose=verbose)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    return model, {"train_accs" : train_accs, "test_accs" : test_accs}


# --
# Run

def _mp_train_worker(models, model_ids, gpu_id, results, verbose=False, **kwargs):
    trainloader, testloader = make_dataloaders()
    for model_id in model_ids:
        t = time()
        
        model, performance = train(models[model_id], trainloader, testloader, **kwargs)
        
        # Save model to disk
        model = model.cpu()
        model_name = model.get_id() + datetime.now().strftime('-%Y%m%d_%H%M%S')
        weight_path = os.path.join('.weights', model_name)
        torch.save(model.state_dict(), weight_path)
        del model
        
        results[model_id] = {
            "model"       : None,
            "weight_path" : weight_path,
            "performance" : performance,
        }
        
        print("worker_train: finished model_id=%d in %f seconds" % (model_id, time() - t), file=sys.stderr)


def train_mp(models, num_gpus=2, **kwargs):
    manager = mp.Manager()
    results = manager.dict()
    
    chunks = np.array_split(list(models.keys()), num_gpus)
    
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        kwargs.update({
            "models"     : models,
            "model_ids"  : chunk,
            "gpu_id"     : gpu_id, 
            "results"    : results,
        })
        p = mp.Process(target=_mp_train_worker, kwargs=kwargs)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Load state dicts from disk
    results = dict(results)
    for k in models.keys():
        models[k].load_state_dict(torch.load(results[k]['weight_path']))
        results[k]['model'] = models[k]
    
    del models
    
    return results

# --
# Make children

# Init
num_neighbors = 8
num_morphs = 5
num_steps = 5
num_epoch_neighbors = 17
num_epoch_final = 100
all_models = {}

set_seeds(123)
base_model = SeaNet({
    1 : (mm.MorphBCRLayer(3, 64, kernel_size=3, padding=1), 0),
    2 : (nn.MaxPool2d(2), 1),
    3 : (mm.MorphBCRLayer(64, 128, kernel_size=3, padding=1), 2),
    4 : (nn.MaxPool2d(2), 3),
    5 : (mm.MorphBCRLayer(128, 128, kernel_size=3, padding=1), 4),
    6 : (mm.MorphFlatLinear(2048 * 4, 10), 5),
}, input_shape=(3, 32, 32))

all_models[-1] = train_mp({0 : base_model}, epochs=10, verbose=True)
best_model = copy.deepcopy(all_models[-1][0]['model'])

# Create a population of neighbors
neibs = {}
neibs[0] = best_model
while len(neibs) < num_neighbors:
    try:
        neibs[len(neibs)] = do_random_morph(best_model, n=num_morphs)
    except:
        pass
    
    print(''.join(['-'] * 100), file=sys.stderr)

# Train
for k, neib in neibs.items():
    neib_model = neib['model'].cuda()
    neibs[k] = train(neib_model, trainloader, testloader, epochs=num_epoch_neighbors, lr_init=lr_init)
    neibs[k]['model'] = neibs[k]['model'].cpu()

hist[len(hist)] = neibs

best_id = np.argmax([o['test_accs'][-1] for k,o in neibs.items()])
best_model = copy.deepcopy(neibs[best_id]['model'])

#


from rsub import *
from matplotlib import pyplot as plt

_ = plt.plot(orig_obj['train_accs'], alpha=1.)
for k, obj in neibs.items():
    _ = plt.plot(obj['train_accs'], label=k, alpha=0.25)

show_plot()
