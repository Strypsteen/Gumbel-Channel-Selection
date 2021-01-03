# EEG Channel Selection with Gumbel-softmax

## About

This Python project is the PyTorch implementation of a concrete EEG channel selection layer based on the Gumbel-softmax method. This layer can be placed in front of any deep neural network architecture to jointly learn the optimal subset of EEG channels for the given task and the network weights. This layer of composed of selection neurons, that each use a continuous relaxation of a discrete distribution across the input channels to learn the optimal one-hot weight vector to select input channels instead of linearly combining them.

## Installation

To install, use pip install -r requirements.txt

## Usage

This implementation operates on the dataset described in [1]. To download this data, follow the instructions at https://github.com/robintibor/high-gamma-dataset and place it in the Data folder. The code can then be run with python selectNchannels.py


 ## References
 
[1] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, “Deep learning with convolutional neural networks for EEG decoding and visualization,” Human brain mapping, vol. 38, no. 11, pp. 5391– 5420, 2017.
