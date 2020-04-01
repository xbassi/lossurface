import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 


class Mark001(nn.Module):

	def __init__(self,op_szie):

		super(Mark001, self).__init__()

		self.lr = nn.LeakyReLU()
		self.pool = nn.MaxPool2d(2, 2)

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
		self.bn1 = nn.BatchNorm2d(64)

		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
		self.bn2 = nn.BatchNorm2d(128)

		self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)
		self.bn3 = nn.BatchNorm2d(128)



		self.fc1 = nn.Linear(128,256)
		self.fbn1 = nn.BatchNorm1d(256)

		self.fc2 = nn.Linear(256,128)
		self.fbn2 = nn.BatchNorm1d(128)
		
		self.fc3 = nn.Linear(128,op_szie)


	def forward(self, x1):

		x1 = self.lr(self.bn1(self.conv1(x1)))
		# print(x1.shape)
		x1 = self.pool(x1)
		x1 = self.lr(self.bn2(self.conv2(x1)))
		# print(x1.shape)
		x1 = self.pool(x1)
		x1 = self.lr(self.bn3(self.conv3(x1)))
		# x1 = self.pool(x1)

		# print(x1.shape)

		x1 = x1.view(x1.shape[0],-1)
		# print(x1.shape)
		

		x1 = self.lr(self.fbn1(self.fc1(x1)))
		x1 = self.lr(self.fbn2(self.fc2(x1)))
		x1 = (self.fc3(x1))
		# x1 = F.softmax(x1,dim=1)

		return x1



