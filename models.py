import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 


class Mark001(nn.Module):

	def __init__(self,op_szie):

		super(Mark001, self).__init__()

		self.lr = nn.LeakyReLU()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
		self.bn1 = nn.BatchNorm2d(16)

		self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(16)

		self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1)
		self.bn3 = nn.BatchNorm2d(16)

		self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1)
		self.bn4 = nn.BatchNorm2d(16)

		# self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1)
		# self.bn5 = nn.BatchNorm2d(16)

		# self.conv6 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
		# self.bn6 = nn.BatchNorm2d(32)

		# self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
		# self.bn7 = nn.BatchNorm2d(32)

		# self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
		# self.bn8 = nn.BatchNorm2d(32)

		# self.conv9 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
		# self.bn9 = nn.BatchNorm2d(16)

		self.fc3_1 = nn.Linear(256,op_szie)
		self.fbn3_1 = nn.BatchNorm1d(op_szie)


	def forward(self, x1):

		x1 = self.lr(self.bn1(self.conv1(x1)))
		# print(x1.shape) 
		x1 = self.lr(self.bn2(self.conv2(x1)))
		# print(x1.shape)
		x1 = self.lr(self.bn3(self.conv3(x1)))
		# print(x1.shape)
		x1 = self.lr(self.bn4(self.conv4(x1)))
		# print(x1.shape)
		# x1 = self.lr(self.bn5(self.conv5(x1)))
		# x1 = self.lr(self.bn6(self.conv6(x1)))
		# x1 = self.lr(self.bn7(self.conv7(x1)))
		# x1 = self.lr(self.bn8(self.conv8(x1)))
		# x1 = self.lr(self.bn9(self.conv9(x1)))

		x1 = x1.view(x1.shape[0],-1)
		# print(x1.shape)
		
		x1 = self.lr(self.fbn3_1(self.fc3_1(x1)))
		x1 = F.softmax(x1,dim=1)

		return x1



