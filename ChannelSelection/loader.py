import torch
import numpy as np
import os


def all_subject_loader_HGD(batch_size,train_split,path):
	
	num_subjects = 14

	#Create dataset
	tr_ds=[]
	val_ds=[]
	test_ds=[]

	for k in range(num_subjects):
		#Load training data
		traindatapath = os.path.join(path,str(k+1)+"traindata.npy")
		trainlabelpath = os.path.join(path,str(k+1)+"trainlabel.npy")
		train_eeg_data = torch.Tensor(np.load(traindatapath))
		train_labels = torch.LongTensor(np.load(trainlabelpath))

		split = round(train_split*train_eeg_data.size(0))

		for i in range(train_eeg_data.size(0)):
			x = train_eeg_data[i,:,:]
			x=x.view(1,x.size(0),x.size(1))
			y = train_labels[i]
			if(i<=split):
				tr_ds.append([x,y])
			else:
				val_ds.append([x,y])

		#Load test data
		testdatapath = path + str(k+1)+"testdata.npy"
		testlabelpath = path + str(k+1)+"testlabel.npy"
		test_eeg_data = torch.Tensor(np.load(testdatapath))
		test_labels = torch.LongTensor(np.load(testlabelpath))

		for i in range(test_eeg_data.size(0)):
			x = test_eeg_data[i,:,:]
			x=x.view(1,x.size(0),x.size(1))
			y = test_labels[i]
			test_ds.append([x,y])

	trainloader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size,
										  shuffle=True)
	valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
										  shuffle=False)
	testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
										  shuffle=False)
	return trainloader,valloader,testloader

def within_subject_loader_HGD(subject,batch_size,train_split,path):

	traindatapath = path + str(subject)+"traindata.npy"
	trainlabelpath = path + str(subject)+"trainlabel.npy"
	train_eeg_data = torch.Tensor(np.load(traindatapath))
	train_labels = torch.LongTensor(np.load(trainlabelpath))

	tr_ds=[]
	val_ds = []
	split = round(train_split*train_eeg_data.size(0))
	for i in range(train_eeg_data.size(0)):
		x = train_eeg_data[i,:,:]
		#x=x[::2,:]
		x=x.view(1,x.size(0),x.size(1))
		y = train_labels[i]
		if(i<= split):
			tr_ds.append([x,y])
		else:
			val_ds.append([x,y])


	testdatapath = path + str(subject)+"testdata.npy"
	testlabelpath = path + str(subject)+"testlabel.npy"
	test_eeg_data = torch.Tensor(np.load(testdatapath))
	test_labels = torch.LongTensor(np.load(testlabelpath))

	test_ds=[]
	for i in range(test_eeg_data.size(0)):
		x = test_eeg_data[i,:,:]
		#x=x[::2,:]
		x=x.view(1,x.size(0),x.size(1))
		y = test_labels[i]
		test_ds.append([x,y])


	trainloader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size,
										  shuffle=False)
	valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
										  shuffle=False)
	testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
										  shuffle=False)
	return trainloader,valloader,testloader
