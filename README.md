# Convolutional Spiking Neural Networks for Intent Detection Based on Anticipatory Brain Potentials Using Electroencephalogram

## Authors: Nathan Lutes, Venkata Sriram Siddhardh Nadendla, K. Krishnamurthy

This repository contains the accompanying code to the paper: "Convolutional Spiking Neural Networks for Detecting Anticipatory Brain Potentials Using Electroencephalogram". The python code was used to generate the results detailed in the paper. Tables 1 and 3 were generated using the KFoldCrossValExp1.py script and table 2 (ablation study) was generated using the exp1_ablation_study.py script. The Functions.py script contains the training and testing code for each model and tools.py contains various utility functions for loading the data etc. The data loaded was pre-processed according to the paper with the padding and normalization steps being performed during the data loading utilities inside of tools.py. define_paths.py is a simple organizational script housing the paths to the data and to the destination where the scripts should save the results. The analysis was performed using the high performance cluster (HPC) resource available for students and faculty at Missouri University of Science and Technology. 

Read the paper here at: https://arxiv.org/abs/2208.06900

