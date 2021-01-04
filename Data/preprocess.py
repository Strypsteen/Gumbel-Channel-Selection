import logging
import sys
import os.path
from collections import OrderedDict
import numpy as np

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
import torch.nn.functional as F
import torch as th
from torch import optim
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
	RuntimeMonitor, CroppedTrialMisclassMonitor

from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.trial_segment import \
	create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize

from braindecode.datasets.sensor_positions import (
	CHANNEL_10_20_APPROX,
	get_channelpos,
)

import scipy.io as sio
import mne


log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def load_bbci_data(filename, low_cut_hz):
	load_sensor_names = None
	loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)


	log.info("Loading data...")
	cnt = loader.load()

	# Cleaning: First find all trials that have absolute microvolt values
	# larger than +- 800 inside them and remember them for removal later
	log.info("Cutting trials...")

	marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
							  ('Rest', [3]), ('Feet', [4])])
	clean_ival = [0, 4000]

	set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
												  clean_ival)

	clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

	log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
		np.sum(clean_trial_mask),
		len(set_for_cleaning.X),
		np.mean(clean_trial_mask) * 100))

	# now pick only sensors with C in their name
	# as they cover motor cortex
	C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
				 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
				 'C6',
				 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
				 'FCC5h',
				 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
				 'CPP5h',
				 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
				 'CCP1h',
				 'CCP2h', 'CPP1h', 'CPP2h']

	cnt = cnt.pick_channels(C_sensors)

	# Further preprocessings
	log.info("Resampling...")
	cnt = resample_cnt(cnt, 250.0)

	print("REREFERENCING")

	log.info("Highpassing...")
	cnt = mne_apply(lambda a: highpass_cnt(a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),cnt)
	log.info("Standardizing...")
	cnt = mne_apply(lambda a: exponential_running_standardize(a.T, factor_new=1e-3,init_block_size=1000,eps=1e-4).T,cnt)

	# Trial interval, start at -500 already, since improved decoding for networks
	ival = [-500, 4000]

	dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)

	dataset.X = dataset.X[clean_trial_mask]
	dataset.y = dataset.y[clean_trial_mask]
	return dataset.X, dataset.y


low_cut_hz=4
for i in range(14):
	print("Start Train Data Subject " + str(i+1) )
	filename = "train/" + str(i+1) + ".mat"
	savenamedata = "train/" + str(i+1) + "traindata.npy"
	savenamelabel = "train/"+ str(i+1) + "trainlabel.npy"
	X,y = load_bbci_data(filename, low_cut_hz)
	np.save(savenamedata,X)
	np.save(savenamelabel,y)

for i in range(14):
	print("Start Test Data Subject " + str(i+1) )
	filename = "test/" + str(i+1) + ".mat"
	savenamedata = "test/" + str(i+1) + "testdata.npy"
	savenamelabel = "test/"+ str(i+1) + "testlabel.npy"
	X,y = load_bbci_data(filename, low_cut_hz)
	np.save(savenamedata,X)
	np.save(savenamelabel,y)
