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

from torchsummary import summary


def main(args,asmv,counter):

	device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)


	train_loader,test_loader = get_loaders(args)


	num_classes = 10
	
	# model = Mark001(num_classes).to(device)
	
	model = models.resnet18()
	model.fc = nn.Linear(512, num_classes)
	
	# model = nn.DataParallel(model)

	model = model.to(device)
	model.train()

	optimizer = torch.optim.Adam(
	    model.parameters(),
	    lr=args.lr_init,
	    amsgrad=True,
	    # momentum=args.momentum,
	    weight_decay=args.wd
	)

	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9, last_epoch=-1)

	criterion = nn.CrossEntropyLoss()


	# summary(model, (3, 32,32))

	# s = Surface()

	score = 0

	for x in range(args.epochs):

		losses = AverageMeter('Loss', ':.6f')


		for i, (inputs, label) in enumerate(train_loader):

			inputs, label = inputs.to(device), label.to(device)

			output = model(inputs)
			
			loss = criterion(output, label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses.update(loss.item(), inputs.size(0))

			# if i % args.surface_track_freq == 0: 
			# 	s.add(model,loss.item(),score)

			if i % args.print_freq == 0:
				print("\tStep: ",i,losses.__str__())

		# scheduler.step()
		print("Epoch: ",x,losses.__str__())
		score = validate(test_loader,model,criterion,device)
		# s.add(model,loss.item(),score)	
		torch.save(model.state_dict(), f"./{args.model_dir}/r18_10")

	# s.plot()

	asmv.draw(model,evalav,train_loader,test_loader,criterion,0.5,counter)


if __name__=="__main__":

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

	parser.add_argument('--surface_track_freq', type=int, default=50, metavar='N', help='surface track frequency (default: 5)')

	args = parser.parse_args()

	print(args)

	asmv = AsymValley(args.graph_dir)

	for i in range(10):

		main(args,asmv,i)