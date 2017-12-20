#!/usr/bin/env python

"""
    main.py
"""

from __future__ import print_function, division

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from seanet import SeaNet
import morph_ops as mo
import morph_layers as mm

from lr import LRSchedule
from utils import progress_bar
from helpers import set_seeds, to_numpy, colstring

torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outpath', type=str, default='./results/models/delete-me')
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--lr-init', type=float, default=0.05)
    parser.add_argument('--no-cuda', action="store_true")
    
    return parser.parse_args()

args = parse_args()

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

# --
# Helpers

def train(net, opt, lr_scheduler, epoch, trainloader):
    _ = net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    batches_per_epoch = len(trainloader)
    
    for batch_idx, (data, targets) in enumerate(trainloader):
        if not args.no_cuda:
            data, targets = data.double().cuda(), targets.cuda()
        
        data, targets = Variable(data), Variable(targets)
        
        # Set LR
        LRSchedule.set_lr(opt, lr_scheduler(epoch + batch_idx / batches_per_epoch))
        
        opt.zero_grad()
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        opt.step()
        
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total


def test(net, testloader):
    _ = net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(testloader):
        if not args.no_cuda:
            data, targets = data.double().cuda(), targets.cuda()
        
        data, targets = Variable(data, volatile=True), Variable(targets)
        
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total


# --
# Define model

set_seeds(123)
net = SeaNet({
    1 : (mm.MorphBCRLayer(3, 32, kernel_size=3, padding=1), 0),
    2 : (nn.MaxPool2d(2), 1),
    3 : (mm.MorphBCRLayer(32, 32, kernel_size=3, padding=1), 2),
    4 : (nn.MaxPool2d(2), 3),
    5 : (mm.MorphBCRLayer(32, 32, kernel_size=3, padding=1), 4),
    6 : (mm.MorphFlatLinear(2048, 10), 5),
}, input_shape=(3, 32, 32))

if not args.no_cuda:
    net = net.cuda()

# --
# Define optimizer

lr_scheduler = getattr(LRSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs)
opt = torch.optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=5e-4)

# --
# Run

train_accs, test_accs = [], []
for epoch in range(0, args.epochs):
    print("Epoch=%d" % epoch, file=sys.stderr)
    
    train_acc = train(net, opt, lr_scheduler, epoch, trainloader)
    test_acc = test(net, testloader)
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(json.dumps({'train_acc' : train_acc, 'test_acc' : test_acc}))

torch.save(net.state_dict(), args.outpath)



