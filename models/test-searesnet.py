#!/usr/bin/env python

"""
    test-resnet.py
"""

from __future__ import division, print_function

import os
import sys
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

sys.path.append('..')
from utils import progress_bar
from searesnet import SeaResNet



# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='searesnet')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

args = parse_args()

nets = {
    'searesnet' : SeaResNet,
}

# --
# IO

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

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)

if args.train_size < 1:
    train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=args.train_size, random_state=args.seed + 1)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
    )
    
    valloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
    )
else:
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )


testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# --
# Helpers

def do_eval(epoch, dataloader):
    _ = net.eval()
    test_loss, correct, total = 0, 0, 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = Variable(data, volatile=True).cuda(), Variable(targets).cuda()
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total

# --
# Learning rate scheduling

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lr_schedule_step(x, breaks=(150, 250)):
    if x < breaks[0]:
        return 0.1
    elif x < breaks[1]:
        return 0.01
    else:
        return 0.001

def lr_schedule_linear(x, lr_init=args.lr_init, epochs=args.epochs):
    return lr_init * float(epochs - x) / epochs

def lr_schedule_cyclical(x, lr_init=args.lr_init, epochs=args.epochs):
    if x < 1:
        return 0.05
    else:
        return lr_init * (1 - x % 1) * (epochs - np.floor(x)) / epochs

def lr_schedule_cyclical_c(x, lr_init=args.lr_init, epochs=args.epochs):
    if x < 1:
        return 0.05
    else:
        return lr_init * (1 - x % 1)


lr_schedules = {
    "step" : lr_schedule_step,
    "linear" : lr_schedule_linear,
    "cyclical" : lr_schedule_cyclical,
    "cyclical_c" : lr_schedule_cyclical_c,
}

lr_schedule = lr_schedules[args.lr_schedule]

# --
# Define model

net = nets[args.net]().cuda()
print(net, file=sys.stderr)

optimizer = optim.SGD(net.parameters(), lr=lr_schedule(0), momentum=0.9, weight_decay=5e-4)

# --
# Train

for epoch in range(0, args.epochs):
    print("Epoch=%d" % epoch, file=sys.stderr)
    
    _ = net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (data, targets) in enumerate(trainloader):
        data, targets = Variable(data).cuda(), Variable(targets).cuda()
        
        set_lr(optimizer, lr_schedule(epoch + batch_idx / len(trainloader)))
        
        optimizer.zero_grad()
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_acc = float(correct) / total
    val_acc = do_eval(epoch, valloader) if args.train_size < 1 else None
    test_acc = do_eval(epoch, testloader)
    
    print(json.dumps({
        'epoch'     : epoch,
        'train_acc' : train_acc,
        'val_acc'   : val_acc,
        'test_acc'  : test_acc,
    }))
