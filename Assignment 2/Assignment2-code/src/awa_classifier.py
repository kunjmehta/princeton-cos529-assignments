import torch
from torch import nn
import torchvision

from repvgg import create_RepVGG_A2
from resnet import resnet50
# from resnext import resnext101_32x8d

class resnet_awa(nn.Module):
	def __init__(self, num_classes=50):
		super().__init__()
		self.resnet = torchvision.models.resnet50(pretrained=False)
		self.resnet.fc = nn.Sequential(nn.BatchNorm1d(2048), 
									   nn.ReLU(), 
									   nn.Dropout(0.25), 
									   nn.Linear(2048, 85))
								
	def forward(self, x):
		return torch.sigmoid(self.resnet(x))

class repvgg_awa(nn.Module):
	def __init__(self, num_classes=50):
		super().__init__()
		self.repvgg = create_RepVGG_A2()
		self.mlp = nn.Sequential(nn.BatchNorm1d(1408), 
								 nn.ReLU(),
								 nn.Linear(1408, 85))
								
	def forward(self, x):
		*_, repvgg_f = self.repvgg(x)
		output = self.mlp(repvgg_f)

		return torch.sigmoid(output)

