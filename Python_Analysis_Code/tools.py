# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:16:09 2023

Convolutional Spiking Neural Networks for Intent Detection Based on Anticipatory
Brain Potentials Using Electroencephalogram

Authors: Nathan Lutes, V. S. S. Nadendla, K. Krishnamurthy
"""
import glob
import numpy as np
from torch import from_numpy, is_tensor, stack, zeros
from torch.utils.data import Dataset
from sklearn.preprocessing import minmax_scale
from snntorch import utils
from snntorch.spikegen import delta as trans2spike
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#define eegdataset class for CNNs and CSNN    
class EEGDataset_CNN_CSNN(Dataset):
    def __init__(self, dataFilePath, model_name, thresh = 0, chanInd = list(range(19)),
                 num_data_points = None, debug = False):
        #read in data
        self.data_list = []
        self.label_list = []
        self.filepath = dataFilePath
        
        #get filepath contents
        self.fileList = glob.glob(self.filepath + '/*')
        
        self.chanInd = chanInd
        self.debug = debug
        
        if num_data_points is None:
            num_data_points = len(self.fileList)
        self.num_data_points = num_data_points
        
        #loop through data
        if self.debug:
            self.count = 0
        for fileInd in range(self.num_data_points):
            if self.debug:
                self.count += 1
                if self.count >= 100:
                    break
            #read data from file
            data = np.load(self.fileList[fileInd])
            #zero out channels not used
            for chan_idx in range(19):
                if chan_idx not in self.chanInd:
                    data[:,chan_idx] = 0*data[:,chan_idx]

            data = np.transpose(np.concatenate([data, np.zeros([996-np.shape(data)[0],
                                                                np.shape(data)[1]])],0))
            data = minmax_scale(data, axis = 1)
            data = np.reshape(data, [1, np.shape(data)[0], np.shape(data)[1]])
            data = from_numpy(data).float()
            if model_name == "CSNN_Spike_Convert":
                data = trans2spike(data, threshold = thresh)
            self.data_list.append(data)
            
            #append label
            if self.fileList[fileInd][-7:-4] == '1-0':
                if model_name == "CSNN" or model_name == "CSNN_Spike_Convert":
                    label = from_numpy(np.array([0,1], dtype = np.float32))
                else:
                    label = from_numpy(np.array([0,1]))
            else:
                if model_name == "CSNN" or model_name == "CSNN_Spike_Convert":
                    label = from_numpy(np.array([1,0], dtype = np.float32))
                else:
                    label = from_numpy(np.array([1,0]))
                
            self.label_list.append(label)
            
    def __len__(self):
        if self.debug:
            return self.count-1
        else:
            return len(self.fileList)
    
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
            
        sample = [self.data_list[idx], self.label_list[idx]]
        return sample        
    
#define dataset class for gnns
#Create custom dataset class to read in the processed EEG data
from spektral import data
from spektral.data import graph
class EEGDataset_GNN(data.Dataset):
    """"
    A custom dataset that reads in the EEG data
    """
    def __init__(self, filepath, chanInd = list(range(19)), debug = 0, 
                 num_data_points = None, **kwargs):
        self.filepath = filepath
        self.chanInd = chanInd
        self.debug = debug
        self.num_data_points = num_data_points
        super().__init__(**kwargs)
        
    def read(self):
        #create storage
        output = []
        PCCArray = np.array([])
        self.fileList = glob.glob(self.filepath + '/*')
        if self.num_data_points is None:
            self.num_data_points = len(self.fileList)
        #organize data into numpy arrays
        if self.debug:
            self.num_data_points = 100
        count = 0
        for file in self.fileList:
            count += 1
            if count > self.num_data_points:
                break
            #create numpy array from csv
            data = np.load(file)
            #drop channels not used
            data = data[:,self.chanInd]
            #transform data
            data = minmax_scale(np.transpose(np.concatenate([data,np.zeros([996-np.shape(data)[0],
                                                     np.shape(data)[1]])],0)), axis = 1)
            #add to PCCarray
            if PCCArray.size == 0:
                PCCArray = np.transpose(data)
            else:
                PCCArray = np.concatenate((PCCArray, np.transpose(data)))
                
            #append to output
            if file[-7:-4] == '1-0':
                output.append(graph.Graph(x = data, y = 1))
            else:
                output.append(graph.Graph(x = data, y = 0))
            self.output = output
            
                               
        #calculate PCC matrix
        self.PCC = np.corrcoef(PCCArray,rowvar=False)
        #calculate absolute value PCC
        absPCC = np.abs(self.PCC)
        #adjacency matrix
        A = absPCC - np.identity(absPCC.shape[0])
        self.a = A
                 
        return output
    
def Load_exp1_data_EEGNet(filePath, one_hot = False, chanInd = list(range(19)),
                          num_data_points = None, debug = False):
    from os.path import join
    from glob import glob
    from sklearn.preprocessing import minmax_scale
    from tensorflow.data import Dataset
    ## this function loads the experiment 1 data
    
    #create storage
    data_list = []
    label_list = []
    #get directory contents
    exp1_data = glob(join(filePath,'*.npy'))
    
    if num_data_points is None:
        num_data_points = len(exp1_data)

    #loop through data
    if debug:
        count = 0
    for fileInd in range(num_data_points):
        if debug:
            count += 1
            if count >= 100:
                break
        # read data from file
        data = np.load(exp1_data[fileInd])
        #zero out channels not used
        for chan_idx in range(19):
            if chan_idx not in chanInd:
                data[:,chan_idx] = 0*data[:,chan_idx]
        
        #transform data
        data = np.transpose(np.concatenate([data,np.zeros([996-np.shape(data)[0],
                                                 np.shape(data)[1]])],0))
        data = minmax_scale(data, axis = 1)
        data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1],1])
        data = data.astype(np.float64)
        
        #save data to list
        data_list.append(data)
        
        #get label
        if exp1_data[fileInd][-7:-4] == '1-0':
            if one_hot:
                label = np.array([0,1])
            else:
                label = np.float32(1.)
        else:
            if one_hot:
                label = np.array([1,0])
            else:
                label = np.float32(0.)
        
        #save label to list
        label_list.append(label)
    
    # change into tensorflow dataset
    data_array = np.array(data_list)
    label_array = np.array(label_list)
    exp1_dataset = Dataset.from_tensor_slices((data_array, label_array))
    
    return exp1_dataset, len(exp1_data)


#create class weights
def calc_class_weights(dataset, one_hot, is_graph = False):
    
    #calculate class weight from dataset
    lengthDataset = len(dataset)
    sum1 = 0
    sum2 = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        if is_graph:
            if sample.y == 1:
                sum2 += 1
            else:
                sum1 += 1
        else:
            if one_hot:
                if sample[1][1] == 1:
                    sum2 += 1
                else:
                    sum1 += 1
            else:
                if sample[1] == 1:
                    sum2 += 1
                else:
                    sum1 += 1
            
    classWeight1 = lengthDataset/(2*sum1)
    classWeight2 = lengthDataset/(2*sum2)
    
    class_weights = [classWeight1, classWeight2]
    return class_weights


#calculate class weights
def calc_class_weights_EEGNet(dataset, one_hot):
    from numpy import array
    
    y_targets = array([target.numpy() for _, target in iter(dataset)],
                        dtype = object)
    sum1 = 0
    sum2 = 0
    for y in y_targets:
        if one_hot:
            if y[1] == 1:
                sum2 += 1
            else:
                sum1 += 1
        else:
            if y == 1:
                sum2 += 1
            else:
                sum1 += 1
    if sum1 > 0:        
      classWeight1 = len(y_targets)/(1*sum1)
    else:
      classWeight1 = 0
    if sum2 > 0:
      classWeight2 = len(y_targets)/(2*sum2)
    else:
      classWeight2 = 0
    class_weights = [classWeight1, classWeight2]
    
    return class_weights


#sample weights
def calc_sample_weights(targets, class_weights):
    #define sample weights
    sample_weights = np.ones(np.size(targets))
    for i in range(len(targets)):
        if targets[i] == 0:
            sample_weights[i] = class_weights[0]
        else:
            sample_weights[i] = class_weights[1]
    return sample_weights

def calc_sample_weights_EEGNet(labels, class_weights, one_hot):
    from numpy import zeros
    from tensorflow import convert_to_tensor
    
    sample_weights = zeros(shape = [len(labels)])
    count = -1
    for label in labels:
        count += 1
        if one_hot:
            if label[1] == 1:
                sample_weights[count] = class_weights[1]
            else:
                sample_weights[count] = class_weights[0]
        else:
            if label == 1:
                sample_weights[count] = class_weights[1]
            else:
                sample_weights[count] = class_weights[0]
    sample_weights = convert_to_tensor(sample_weights)
    
    return sample_weights


#get list of labels
def get_label_list(dataset, is_graph = False):
    y=[]
    for i in range(len(dataset)):
        if is_graph:
            y.append(dataset[i].y)
        else:
            if len(dataset[i][1]) == 1:
                y.append(dataset[i][1])
            else:
                y.append(dataset[i][1][1])
        
    return y


#get labels for EEGNet
def get_labels_EEGNet(dataset, one_hot):
    y = []
    dataset = dataset.batch(1)
    for _, target in dataset:
        if one_hot:
            y.append(target[0][1])
        else:
            y.append(target)
        
    return y

#split dataset according to indices
def get_split_dataset_EEGNet(ds, indices):
    from tensorflow import constant, int64
    from tensorflow.math import reduce_any
    
    x_indices = constant(indices, dtype = int64)
    
    def is_index_in(index, rest):
        return reduce_any(index == x_indices)
    
    def drop_index(index, rest):
        return rest

    selected_ds = ds \
        .enumerate() \
        .filter(is_index_in) \
        .map(drop_index)
    return selected_ds


#network forward pass
def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return stack(spk_rec), stack(mem_rec)


#calculate loss
def calc_loss(mem_rec, targets, num_steps, loss_fn, dtype, myDevice):
    
    loss_val = zeros((1), dtype=dtype, device=myDevice)
    for step in range(num_steps):
        loss_val += loss_fn(mem_rec[step], targets)
        
    return loss_val

#early stopping function
def early_stop_calc(tot_loss, miniBatchIter, smallest_loss, wait, patience):
    
    avg_loss = tot_loss/miniBatchIter
    if float(avg_loss) < smallest_loss:
        smallest_loss = float(avg_loss)
        wait = 0
    else:
        wait += 1
      
    if wait == patience:
        break_from_loop = 1
    else:
        break_from_loop = 0
  
    return smallest_loss, wait, break_from_loop

def get_metrics(predicted, targets):
    
    #most important metrics are:
        #not labeling zero's as ones: minimize false positive rate
        #not labeling one's as zeros: minimize false negative rate
        #true positive rate
        #true negative rate
        #f1 score
    
    acc = accuracy_score(targets, predicted)
    f1 = f1_score(targets, predicted)
    tn, fp, fn, tp = confusion_matrix(targets, predicted).ravel()
    #compute other metrics from confusion matrix
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn) #true positive rate
    FPR = fp / (fp + tn)
    TNR = tn / (tn + fp)
    FNR = fn / (fn + tp)
    
    
    return (acc, f1, precision, recall, FPR, TNR, FNR)
    