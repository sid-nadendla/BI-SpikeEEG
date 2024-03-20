# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:21:56 2023

Convolutional Spiking Neural Networks for Intent Detection Based on Anticipatory
Brain Potentials Using Electroencephalogram

Authors: Nathan Lutes, V. S. S. Nadendla, K. Krishnamurthy
"""

def main():
    #this function defines the paths for data loading and storing
    import os
    
    #data path
    dataFilePath = r'C:\Users\nalmrb\AppData\Roaming\MobaXterm\home\Exp1Data'
    
    #results path
    results_path = r'C:\Users\nalmrb\Desktop\ExperimentFiles\Project1_debug'
    if os.path.exists(results_path) is False:
        os.mkdir(results_path)
    
    #GNN model paths
    GCN_save_path = os.path.join(results_path, 'GCN')
    if os.path.exists(GCN_save_path) is False:
        os.mkdir(GCN_save_path)
    GCS_save_path = os.path.join(results_path, 'GCS')
    if os.path.exists(GCS_save_path) is False:
       os.mkdir(GCS_save_path)
    GIN_save_path = os.path.join(results_path, 'GIN')
    if os.path.exists(GIN_save_path) is False:
        os.mkdir(GIN_save_path)
    
    paths = {'dataFilePath':dataFilePath, 'results_path':results_path, 
             'GCN_save_path':GCN_save_path, 'GCS_save_path':GCS_save_path,
             'GIN_save_path':GIN_save_path}
    
    return paths
    
#run main function of script if running from top-level
if __name__ == '__main__':
    import sys
    sys.exit(main())