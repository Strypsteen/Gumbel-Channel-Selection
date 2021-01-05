import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from loader import within_subject_loader_HGD, all_subject_loader_HGD
from models import SelectionNet, init_weights

import statistics
from random import randint
import importlib


parser = argparse.ArgumentParser(description='PyTorch Channel Selection Training')

parser.add_argument('--M',type=int,default=3,

					help='number of selection neurons')

parser.add_argument('--epochs',type=int, default=200,

					help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', type=int, default=16, 

					help='mini-batch size')

parser.add_argument('--gradacc', type = int, default=1,

					help='gradient accumulation')

parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4, 

					help='weight decay')

parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, 

					help='initial learning rate')

parser.add_argument('--lamba', type=float, default=0.1, 

					help='regularization weight')

parser.add_argument('--start_temp',type=float,default=10.0,

					help='initial temperature')
parser.add_argument('--end_temp',type=float,default=0.1,

					help='final temperature')

parser.add_argument('--train_split',type=float,default=0.8,

					help='training-validation data split')
parser.add_argument('--patience', type=int, default=10,

					help='amount of epochs before early stopping')

parser.add_argument('--stop_delta', type=float, default=1e-3,

					help='maximal drop in validation loss for early stopping')

parser.add_argument('--entropy_lim', type=float, default=0.05,

					help='mean entropy for the selection neurons to be reached for convergence')

parser.add_argument('--seed',type=int,default=0,

					help='random seed, 0 indicates randomly chosen seed')

parser.add_argument('-v', action="store_true", default=True, dest="verbose")


def main():

	global args,enable_cuda
################################################################ INIT #################################################################################
	
	args = parser.parse_args()

	cwd=os.getcwd()
	dpath=os.path.dirname(cwd)
	#Paths for data, model and checkpoint
	data_path = os.path.join(dpath,'Data/')
	model_save_path = os.path.join(dpath,'Models','Model_GumbelregHighgamma_M'+str(args.M))
	checkpoint_path = os.path.join(dpath,'Models','Checkpoint_GumbelregHighgamma_M'+str(args.M))
	if not os.path.isdir(os.path.join(dpath,'Models')):
		os.makedirs(os.path.join(dpath,'Models'))

	#Check if CUDA is available
	enable_cuda = torch.cuda.is_available()
	if(args.verbose):
		print('GPU computing: ', enable_cuda)

	#Set random seed
	if(args.seed==0):
		args.seed=randint(1,99999)

	#Initialize devices with random seed
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	training_accs = []
	val_accs=[]
	test_accs = []

	#Create a vector of length epochs, decaying start_value to end_value exponentially, reaching end_value at end_epoch
	def exponential_decay_schedule(start_value,end_value,epochs,end_epoch):
		t = torch.FloatTensor(torch.arange(0.0,epochs))
		p = torch.clamp(t/end_epoch,0,1)
		out = start_value*torch.pow(end_value/start_value,p)

		return out

	#Network loss function
	def loss_function(output,target,model,lamba,weight_decay):
		l = nn.CrossEntropyLoss()
		sup_loss = l(output,target)
		reg = model.regularizer(lamba,weight_decay)

		return sup_loss,reg

	#Create schedule for temperature and regularization threshold
	temperature_schedule = exponential_decay_schedule(args.start_temp,args.end_temp,args.epochs,int(args.epochs*3/4))
	thresh_schedule = exponential_decay_schedule(3.0,1.1,args.epochs,args.epochs)

	#Load data
	num_subjects = 14
	input_dim=[44,1125]
	train_loader,val_loader,test_loader = all_subject_loader_HGD(batch_size=args.batch_size,train_split=args.train_split,path=data_path)


################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################

	if(args.verbose):
		print('Start training')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	#Instantiate model
	model = SelectionNet(input_dim,args.M)
	if(enable_cuda):
		model.cuda()
	model.set_freeze(False)

	optimizer = torch.optim.Adam(model.parameters(),args.lr)

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0

	while epoch in range(args.epochs) and (not early_stop):

		#Update temperature and threshold
		model.set_thresh(thresh_schedule[epoch])
		model.set_temperature(temperature_schedule[epoch])

		#Perform training step
		train(train_loader, model, loss_function, optimizer,epoch,args.weight_decay,args.lamba,args.gradacc,args.verbose)
		val_loss = validate(val_loader,model,loss_function,epoch,args.weight_decay,args.lamba,args.verbose)
		tr_acc,val_acc,test_acc=test(train_loader,val_loader,test_loader,model,loss_function,args.weight_decay,args.verbose)

		#Extract selection neuron entropies, current selections and probability distributions
		H,sel,probas = model.monitor()

		#If selection convergence is reached, enable early stopping scheme
		if((torch.mean(H.data)<=args.entropy_lim) and (val_loss>prev_val_loss-args.stop_delta)):
			patience_timer+=1
			if(args.verbose):
				print('Early stopping timer ', patience_timer)
			if(patience_timer == args.patience):
				early_stop = True
		else:
			patience_timer=0
			H,sel,probas = model.monitor()
			torch.save(model.state_dict(),checkpoint_path)
			prev_val_loss = val_loss


		epoch+=1

	if(args.verbose):
		print('Channel selection finished')

	#Store subject independent model
	model.load_state_dict(torch.load(checkpoint_path))
	pretrained_path = str(model_save_path+'all_subjects_channels_selected.pt')
	torch.save(model.state_dict(), pretrained_path) 

################################################################ SUBJECT FINETUNING  #################################################################################

	if(args.verbose):
		print('Start subject specific training')

	for k in range(1,num_subjects+1):


		if(args.verbose):
			print('Start training for subject ' + str(k))

		torch.manual_seed(args.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		#Load subject independent model and freeze selection neurons
		model = SelectionNet(input_dim,args.M)
		model.load_state_dict(torch.load(pretrained_path))
		if(enable_cuda):
			model.cuda()
		model.set_freeze(True)

		#Load subject dependent data
		train_loader,val_loader,test_loader = within_subject_loader_HGD(subject=k,batch_size=args.batch_size,train_split=args.train_split,path=data_path)
	
		optimizer = torch.optim.Adam(model.parameters(),args.lr)

		prev_val_loss = 100
		patience_timer = 0
		early_stop = False
		epoch = 0
		while epoch in range(args.epochs) and (not early_stop):

			#Perform train step
			train(train_loader, model, loss_function, optimizer,epoch,args.weight_decay,args.lamba,args.gradacc,args.verbose)
			val_loss = validate(val_loader,model,loss_function,epoch,args.weight_decay,args.lamba,args.verbose)
			tr_acc,val_acc,test_acc=test(train_loader,val_loader,test_loader,model,loss_function,args.weight_decay,args.verbose)

			#Extract selection neuron entropies, current selections and probability distributions   
			H,sel,probas = model.monitor()

			#Perform early stopping
			if(val_loss>prev_val_loss-args.stop_delta):
				patience_timer+=1
				if(args.verbose):
					print('Early stopping timer ', patience_timer)
				if(patience_timer == args.patience):
					early_stop = True
			else:
				patience_timer=0
				torch.save(model.state_dict(),checkpoint_path)
				prev_val_loss = val_loss

			epoch+=1


		#Store model with lowest validation loss
		model.load_state_dict(torch.load(checkpoint_path))    
		path = str(model_save_path+'finished_subject'+str(k)+'.pt')
		torch.save(model.state_dict(), path)
			
		#Evaluate model
		tr_acc,val_acc,test_acc = test(train_loader,val_loader,test_loader,model,loss_function,args.weight_decay,args.verbose)
		training_accs.append(tr_acc)
		val_accs.append(val_acc)
		test_accs.append(test_acc)

 ################################################################ TERMINATION  #################################################################################           

	print('Selection', sel.data)
	print('Training accuracies', training_accs)
	print('Validation accuracies', val_accs)
	print('Testing accuracies', test_accs)

	tr_med = statistics.median(training_accs)
	val_med = statistics.median(val_accs)
	test_med = statistics.median(test_accs)
	tr_mean = statistics.mean(training_accs)
	val_mean = statistics.mean(val_accs)
	test_mean = statistics.mean(test_accs)

	print('Training median accuracy', tr_med)
	print('Validation median accuracy', val_med)
	print('Testing median accuracy', test_med)
	print('Training mean accuracy', tr_mean)
	print('Validation mean accuracy', val_mean)
	print('Testing mean accuracy', test_mean)

#train 1 epoch
def train(train_loader, model, loss_function, optimizer, epoch, weight_decay,lamba,gradacc,verbose):

	global running_loss, running_sup_loss, running_reg, running_acc,enable_cuda

	model.train()

	for i, (data, labels) in enumerate(train_loader):

		if(enable_cuda):
			data= data.cuda()
			labels = labels.cuda()

		if(i==0):
			running_loss = 0.0
			running_reg = 0.0
			running_sup_loss = 0.0
			running_acc = np.array([0,0])

		output = model(data)

		sup_loss,reg = loss_function(output, labels, model,lamba,weight_decay)
		loss = sup_loss + reg
		loss=loss/gradacc

		loss.backward()

		#Perform gradient accumulation
		if((i+1)%gradacc ==0):
			optimizer.step()
			optimizer.zero_grad()

		#running accuracy
		score, predicted = torch.max(output,1)
		total = predicted.size(0)
		correct = (predicted == labels).sum().item()
		running_acc = np.add(running_acc, np.array([correct,total]))

		# print statistics
		running_loss += loss.item()
		running_reg += reg.item()
		running_sup_loss += sup_loss.item()
		N = len(train_loader)
		if(i==N-1):
			if(verbose):
				print('[%d, %5d] loss: %.3f acc: %d %% supervised loss: %.3f regularization loss %.3f'%
						(epoch + 1, i + 1, running_loss / N, 100*running_acc[0]/running_acc[1], running_sup_loss/N, running_reg/N))
			running_loss = 0.0
			running_reg = 0.0
			running_sup_loss = 0.0
			running_acc = (0,0)

def validate(val_loader,model,loss_function,epoch,weight_decay,lamba,verbose):

	global val_acc,val_loss,enable_cuda

	with torch.no_grad():
		model.eval()

		for i, (data, labels) in enumerate(val_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			if(i==0):
				val_loss = 0.0
				val_acc = np.array([0,0])

			output = model(data)
			sup_loss,reg = loss_function(output, labels, model,lamba,weight_decay)
			loss = sup_loss

			#running accuracy
			score, predicted = torch.max(output,1)
			total = predicted.size(0)
			correct = (predicted == labels).sum().item()
			val_acc = np.add(val_acc, np.array([correct,total]))

			# print statistics
			val_loss += loss.item()
			N = len(val_loader)
			if(i == N-1):
				if(verbose):
					print('[%d, %5d] Validation loss: %.3f Validation accuracy: %d %%'%
						(epoch + 1, i + 1, val_loss / N,100*val_acc[0]/val_acc[1] ))

	return val_loss/N

def test(train_loader,val_loader,test_loader, model,loss_function,weight_decay,verbose):

	global enable_cuda

	with torch.no_grad():

		model.train()

		total = 0
		correct = 0

		for i, (data, labels) in enumerate(train_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			output = model(data)
			score, predicted = torch.max(output,1)
			total += predicted.size(0)
			correct += (predicted == labels).sum().item()

		tr_acc = correct/total

		if(verbose):
			print('Training accuracy: %d %%' % (100 * tr_acc))

		model.eval()

		total = 0
		correct = 0

		for i, (data, labels) in enumerate(val_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			output = model(data)
			score, predicted = torch.max(output,1)
			total += predicted.size(0)
			correct += (predicted == labels).sum().item()

		val_acc = correct/total

		if(verbose):
			print('Validation accuracy: %d %%' % (100 * val_acc))

		total=0
		correct=0

		for i, (data, labels) in enumerate(test_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			output = model(data)
			score, predicted = torch.max(output,1)
			total += predicted.size(0)
			correct += (predicted == labels).sum().item()

		test_acc = correct/total

		if(verbose):
			print('Test accuracy: %d %%' % (100 * test_acc))

		return tr_acc,val_acc,test_acc

if __name__ == '__main__':

	main()
