import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import sys
import argparse

from utils import AverageMeter,Surface,AsymValley

from models import Mark001
from torchvision import models

from data import get_loaders

from val import validate,evalav

parser = argparse.ArgumentParser(description='SGD-FastConv training')


parser.add_argument('--data_dir', type=str, default="./data/", required=False, help='training directory (default: None)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size (default: 32)')
parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='number of workers (default: 4)')

parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr_init', type=float, default=0.001, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-6, help='weight decay (default: 1e-4)')

parser.add_argument('--print_freq', type=int, default=500, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--save_freq', type=int, default=1, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--model_dir', type=str, default="checkpoints", required=False, help='Model Save Directory')

parser.add_argument('--graph_dir', type=str, default="./graphs/cifar10_r10", required=False, help='Graph Save Directory')


args = parser.parse_args()
print(args)


device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


train_loader,test_loader = get_loaders(args)


asmv = AsymValley(args.graph_dir)


num_classes = 10

model = models.resnet18()
model.fc = nn.Linear(512, num_classes)

model.load_state_dict(torch.load("./checkpoints/r18_10"))

criterion = nn.CrossEntropyLoss()

model = model.to(device)

asmv.draw(model,evalav,train_loader,test_loader,criterion,1.8,100)


