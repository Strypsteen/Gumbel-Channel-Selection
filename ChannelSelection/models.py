import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
import scipy.io as sio
from torch.nn.parameter import Parameter



##PRUNED TRAINING

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

##SELECT N CHANNELS

def init_weights(m):
	if (type(m) == nn.Linear or type(m) == nn.Conv2d):
		torch.nn.init.xavier_uniform_(m.weight)

class MSFBCNN(nn.Module):
	def __init__(self,input_dim,output_dim,FT=10):
		super(MSFBCNN, self).__init__()
		self.T = input_dim[1]
		self.FT = FT
		self.D = 1
		self.FS = self.FT*self.D
		self.C=input_dim[0]
		self.output_dim = output_dim
		
		# Parallel temporal convolutions
		self.conv1a = nn.Conv2d(1, self.FT, (1, 65), padding = (0,32),bias=False)
		self.conv1b = nn.Conv2d(1, self.FT, (1, 41), padding = (0,20),bias=False)
		self.conv1c = nn.Conv2d(1, self.FT, (1, 27), padding = (0,13),bias=False)
		self.conv1d = nn.Conv2d(1, self.FT, (1, 17), padding = (0,8),bias=False)

		self.batchnorm1 = nn.BatchNorm2d(4*self.FT, False)
		
		# Spatial convolution
		self.conv2 = nn.Conv2d(4*self.FT, self.FS, (self.C,1),padding=(0,0),groups=1,bias=False)
		self.batchnorm2 = nn.BatchNorm2d(self.FS, False)

		#Temporal average pooling
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))

		self.drop=nn.Dropout(0.5)

		#Classification
		self.fc1 = nn.Linear(self.FS*math.ceil(1+(self.T-75)/15), self.output_dim)

	def forward(self, x):

		# Layer 1
		x1 = self.conv1a(x);
		x2 = self.conv1b(x);
		x3 = self.conv1c(x);
		x4 = self.conv1d(x);

		x = torch.cat([x1,x2,x3,x4],dim=1)
		x = self.batchnorm1(x)

		# Layer 2
		x = torch.pow(self.batchnorm2(self.conv2(x)),2)
		x = self.pooling2(x)
		x = torch.log(x)
		x = self.drop(x)
		
		# FC Layer
		x = x.view(-1, self.num_flat_features(x))
		x = self.fc1(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


class SelectionLayer(nn.Module):
	def __init__(self, N,M,temperature=1.0):

		super(SelectionLayer, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
		self.N = N
		self.M = M
		self.qz_loga = Parameter(torch.randn(N,M)/100)

		self.temperature=self.floatTensor([temperature])
		self.freeze=False
		self.thresh=3.0

	def quantile_concrete(self, x):

		g = -torch.log(-torch.log(x))
		y = (self.qz_loga+g)/self.temperature
		y = torch.softmax(y,dim=1)

		return y

	def regularization(self):
		
		eps = 1e-10
		z = torch.clamp(torch.softmax(self.qz_loga,dim=0),eps,1)
		H = torch.sum(F.relu(torch.norm(z,1,dim=1)-self.thresh))

		return H

	def get_eps(self, size):

		eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)

		return eps

	def sample_z(self, batch_size, training):

		if training:

			eps = self.get_eps(self.floatTensor(batch_size, self.N, self.M))
			z = self.quantile_concrete(eps)
			z=z.view(z.size(0),1,z.size(1),z.size(2))
	 
			return z

		else:

			ind = torch.argmax(self.qz_loga,dim=0)
			one_hot = self.floatTensor(np.zeros((self.N,self.M)))
			for j in range(self.M):
					one_hot[ind[j],j]=1
			one_hot=one_hot.view(1,1,one_hot.size(0),one_hot.size(1))
			one_hot = one_hot.expand(batch_size,1,one_hot.size(2),one_hot.size(3))

			return one_hot

	def forward(self, x):

		z = self.sample_z(x.size(0),training=(self.training and not self.freeze))
		z_t = torch.transpose(z,2,3)
		out = torch.matmul(z_t,x)
		return out

class SelectionNet(nn.Module):
	
	def __init__(self,input_dim,M,output_dim=4):
		super(SelectionNet,self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.N = input_dim[0]
		self.T = input_dim[1]
		self.M = M
		self.input_dim = input_dim
		self.output_dim = output_dim
			
		self.network = MSFBCNN(input_dim=[self.M,self.T],output_dim=output_dim)

		self.selection_layer = SelectionLayer(self.N,self.M)

		self.layers = self.create_layers_field()
		self.apply(init_weights)

	def forward(self,x):

		y_selected = self.selection_layer(x)
		out = self.network(y_selected)    

		return out

	def regularizer(self,lamba,weight_decay):

		#Regularization of selection layer
		reg_selection=self.floatTensor([0])
		#L2-Regularization of other layers
		reg=self.floatTensor([0])
		for i,layer in enumerate(self.layers):
			if(type(layer) == SelectionLayer):
				reg_selection += layer.regularization()
			else:
				reg+=torch.sum(torch.pow(layer.weight,2))
		reg = weight_decay*reg + lamba*reg_selection
		return reg

	def create_layers_field(self):
		layers = []
		for idx, m in enumerate(self.modules()):
			if(type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == SelectionLayer):
				layers.append(m)
		return layers

	def get_num_params(self):
		t=0
		for i,layer in enumerate(self.layers):
			print('Layer ' + str(i))
			print(layer)
			n=0
			for p in layer.parameters():
				n += np.prod(np.array(p.size()))
			print('Amount of parameters:' + str(n))
			t+=n
		print('Total amount of parameters ' + str(t))
		return t

	def set_temperature(self,temp):
		m=self.selection_layer
		m.temperature=temp

	def set_thresh(self,thresh):
		m=self.selection_layer
		m.thresh=thresh

	def monitor(self):

		m = self.selection_layer
		eps = 1e-10
		#Probability distributions
		z = torch.clamp(torch.softmax(m.qz_loga,dim=0),eps,1)
		#Normalized entropy
		H = - torch.sum(z*torch.log(z),dim=0)/math.log(self.N)
		#Selections
		s = torch.argmax(m.qz_loga,dim=0)+1

		return H,s,z

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

	def set_freeze(self,x):

		m = self.selection_layer
		if(x):
			for param in m.parameters():
				param.requires_grad=False
			m.freeze = True
		else:
			for param in m.parameters():
				param.requires_grad=True
			m.freeze = False
