import torch

import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
	def __init__(self):
		super(Head, self).__init__()
		self.fc1 = nn.Linear(6,8)
		self.fc2 = nn.Linear(8,8)
		self.fc3 = nn.Linear(8,2)
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return(x)
