# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 19:39:01 2023

Convolutional Spiking Neural Networks for Intent Detection Based on Anticipatory
Brain Potentials Using Electroencephalogram

Authors: Nathan Lutes, V. S. S. Nadendla, K. Krishnamurthy
"""

# train different networks on paper 1 using k fold cross-validation strategy

#imports
import define_paths

#flags
CSNN = 0
CNN = 0
GNN = 0
CSNNSpikeConvert = 0
EEGNet = 1

#parameters
k = 2 #number of folds
batch_size = 8
num_epochs = 3
patience = 50
debug = 1

params = {'k':k, 'batch_size':batch_size, 'num_epochs':num_epochs, 
          'patience':patience, 'debug':debug}

#get paths
paths = define_paths.main()

#run functions
if CSNN:
    from Functions import CSNN_KFoldCrossVal
    CSNN_metrics = CSNN_KFoldCrossVal(paths, params)
    
if CNN:
    from Functions import CNN_KFoldCrossVal
    CNN_metrics = CNN_KFoldCrossVal(paths, params)
    
if GNN:
    from Functions import GNN_KFoldCrossVal
    GNN_metrics = GNN_KFoldCrossVal(paths, params)
    
if CSNNSpikeConvert:
    from Functions import CSNNSpikeConvert_KFoldCrossVal
    CSNNSpikeConvert_metrics = CSNNSpikeConvert_KFoldCrossVal(paths, params)
    
if EEGNet:
    from Functions import EEGNet_KFoldCrossVal
    EEGNet_metrics = EEGNet_KFoldCrossVal(paths, params)
