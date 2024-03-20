# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:27:53 2023

Convolutional Spiking Neural Networks for Intent Detection Based on Anticipatory
Brain Potentials Using Electroencephalogram

Authors: Nathan Lutes, V. S. S. Nadendla, K. Krishnamurthy
"""

# Ablation study: reduce number of channels in loop to determine effects on performance

#imports
import define_paths

#flags
CSNN = 1
CNN = 1
GNN = 1
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

#define channel indices used
# P7,P4,Cz,Pz,P3,P8,O1,O2,T8,F8,C4,F4,Fp2,Fz,C3,F3,Fp1,T7,F7
# try different sets of electrodes starting with set used in khaliliardali etc.
chanInds = [[2,3,10,13,14]];

#get paths
paths = define_paths.main()

#define ablations study path
from os.path import join, exists
from os import mkdir
ablation_study_results_path = join(paths.get('results_path'), 'ablation_study')
if exists(ablation_study_results_path) is False:
    mkdir(ablation_study_results_path)

for loopInd in range(len(chanInds)):
    
    results_path = join(ablation_study_results_path, 
                        f'ablation_study_idx_{loopInd}')
    paths.update({'results_path':results_path})
    if exists(results_path) is False:
        mkdir(results_path)

    chanInd = chanInds[loopInd]
    
    params.update({'chanInd':chanInd})
    
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