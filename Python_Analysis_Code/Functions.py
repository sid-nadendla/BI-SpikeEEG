# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 19:42:53 2023

Convolutional Spiking Neural Networks for Intent Detection Based on Anticipatory
Brain Potentials Using Electroencephalogram

Authors: Nathan Lutes, V. S. S. Nadendla, K. Krishnamurthy
"""

#this script contains the different KFoldCrossVal functions

def CSNN_KFoldCrossVal(paths, params):
    
    import os
    import torch
    from torch import nn as nn
    from torch import device
    from torch.cuda import is_available
    from torch.optim import Adam
    from torch.utils.data import DataLoader, SubsetRandomSampler
    import snntorch as snn
    from snntorch import surrogate
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from time import perf_counter
    from tools import EEGDataset_CNN_CSNN
    from tools import forward_pass
    from tools import calc_class_weights
    from tools import get_label_list
    from tools import calc_loss
    from tools import get_metrics
    from tools import early_stop_calc
    
    #paths
    dataFilePath = paths.get('dataFilePath')
    results_path = paths.get('results_path')

    #parameters
    myDevice = device("cuda") if is_available() else device("cpu")
    k = params.get('k')
    batch_size = params.get('batch_size')
    num_epochs = params.get('num_epochs')
    patience = params.get('patience')
    chanInd = params.get('chanInd')
    if chanInd is None:
        chanInd = list(range(19))
    debug = params.get('debug')
    if debug is None:
        debug = False
    dtype = torch.float
    one_hot = True
    
    # Temporal Dynamics
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 25
    #optimizer parameters
    opt_lr = 5e-4
    opt_betas = (0.9, 0.999)
    
    #set seed
    torch.manual_seed(7)
                
    #instantiate EEG dataset
    dataset = EEGDataset_CNN_CSNN(dataFilePath, model_name = "CSNN",
                                  chanInd = chanInd, debug = debug)

    #calculate class weight from dataset
    class_weights = calc_class_weights(dataset, one_hot)

    # initialize loss
    weights = torch.from_numpy(np.array(class_weights))
    weights = weights.to(myDevice)
    loss_fn = nn.CrossEntropyLoss(weight = weights.float())
    
    #get list of labels
    labels = get_label_list(dataset)

    #Kfold splits
    splits = StratifiedKFold(n_splits = k, shuffle= True, random_state=7) 

    #initiate metrics list
    metrics_list = np.ones(shape = [k,10])

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.zeros(len(dataset)),
                                                            labels)):
        
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=len(val_idx),
                                 sampler=test_sampler)
        
        #initialize model
        net = nn.Sequential(nn.Conv2d(1, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 64, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(15744, 2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(myDevice)
                        
        #optimizer
        optimizer = Adam(net.parameters(), lr=opt_lr, betas=opt_betas)
        
        #training
        smallest_loss = 999999
        wait = 0
        #start timer
        time_ref = perf_counter()
        # Outer training loop
        for epoch in range(num_epochs):    
            tot_loss = 0
            train_batch = iter(train_loader)
        
            # Minibatch training loop
            miniBatchIter = 0
            for data, targets in train_batch:
                miniBatchIter += 1
                
                #send data to device
                data = data.to(myDevice)
                targets = targets.to(myDevice)
        
                # forward pass
                net.train()
                spk_rec, mem_rec = forward_pass(net, num_steps, data)
        
                # initialize the loss & sum over time
                loss_val = calc_loss(mem_rec, targets, num_steps, loss_fn,
                                     dtype, myDevice)
                tot_loss = tot_loss + float(loss_val)
        
                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
        
            #Early stopping
            smallest_loss, wait, break_from_loop = early_stop_calc(tot_loss,
                miniBatchIter, smallest_loss, wait, patience)         
            if break_from_loop:
                break
              
        #stop timer
        time_stop = perf_counter()
        train_time = time_stop - time_ref

        #evaluate
        with torch.no_grad():
            net.eval()
            for data, targets in iter(test_loader):
                data = data.to(myDevice)
                targets = targets.to(myDevice)
                
                # forward pass
                time_ref = perf_counter()
                test_spk, _ = forward_pass(net,num_steps,data)
                time_stop = perf_counter()
                _, predicted = test_spk.sum(dim=0).max(1)
                test_inf_time = time_stop - time_ref
        
        # calculate metrics
        targets_for_metrics = targets[:,1]
        predicted = predicted.cpu()
        targets_for_metrics = targets_for_metrics.cpu()
        metrics = get_metrics(predicted, targets_for_metrics)
        
        #save results
        if fold+1 == 1:
            res = open(os.path.join(results_path,"CSNN_KFoldResults.txt"), 'w')
        else:
            res = open(os.path.join(results_path,"CSNN_KFoldResults.txt"), 'a')
        res.write("Fold: " + str(fold+1) + "\n")
        res.write("Acc = "+ str(metrics[0] * 100) +\
                  ", FPR = " + str(metrics[4] * 100) +\
                      ", TPR = " + str(metrics[3] * 100) +\
                          ", FNR = " + str(metrics[6] * 100) +\
                              ", TNR = " + str(metrics[5] * 100) +\
                                  ", F1 = " + str(metrics[1]) +\
                                      ", epochs = " + str(epoch) +\
                                          ", Train Time = " + str(train_time) +\
                                              ", Test Set Inference Time = " + str(test_inf_time) +'\n')
        res.close()
        
        metrics_list[fold,0:7] = metrics
        metrics_list[fold,7] = epoch
        metrics_list[fold,8] = train_time
        metrics_list[fold,9] = test_inf_time
    
    #calculate grand average
    grand_ave_metrics = np.mean(metrics_list, axis = 0)

    #calculate standard deviation
    grand_std_metrics = np.std(metrics_list, axis = 0)

    #save
    res = open(os.path.join(results_path, 'CSNN_KFoldResults.txt'), 'a')
    res.write('\n')
    res.write("Grand Average:\n")
    res.write("Acc = "+ str(grand_ave_metrics[0] * 100) +\
              ", FPR = " + str(grand_ave_metrics[4] * 100) +\
                  ", TPR = " + str(grand_ave_metrics[3] * 100) +\
                      ", FNR = " + str(grand_ave_metrics[6] * 100) +\
                          ", TNR = " + str(grand_ave_metrics[5] * 100) +\
                              ", F1 = " + str(grand_ave_metrics[1]) +\
                                  ", epochs = " + str(grand_ave_metrics[7]) +\
                                      ", Train Time = " + str(grand_ave_metrics[8]) +\
                                          ", Test Set Inference Time = " + str(grand_ave_metrics[9]) + '\n')
    res.write("Standard Deviation:\n")
    res.write("Acc = "+ str(grand_std_metrics[0] * 100) +\
              ", FPR = " + str(grand_std_metrics[4] * 100) +\
                  ", TPR = " + str(grand_std_metrics[3] * 100) +\
                      ", FNR = " + str(grand_std_metrics[6] * 100) +\
                          ", TNR = " + str(grand_std_metrics[5] * 100) +\
                              ", F1 = " + str(grand_std_metrics[1]) +\
                                  ", epochs = " + str(grand_std_metrics[7]) +\
                                      ", Train Time = " + str(grand_std_metrics[8]) +\
                                          ", Test Set Inference Time = " + str(grand_std_metrics[9]) + '\n')
    res.close()
    
    return metrics_list

def train_GNN(model_name, data, params, paths):
    #imports
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.losses import BinaryCrossentropy
    from spektral.data.loaders import MixedLoader
    from tensorflow.random import set_seed
    from time import perf_counter
    import numpy as np
    from numpy.random import seed
    from tools import get_metrics, calc_sample_weights, early_stop_calc

    seed(10)
    set_seed(10)
    
    #params
    batch_size = params.get('batch_size')
    epochs = params.get('num_epochs')
    patience = params.get('patience')
    class_weights = params.get('class_weights')
    isKfold = params.get('isKfold')
    debug = params.get('debug')
    if debug is None:
        debug = False
    lr = 5e-4
    epsilon = 1e-8
    
    #data
    data_tr = data.get('data_tr')
    data_te = data.get('data_te')
    
    if model_name == "GCN":
        from spektral.layers import GlobalAttnSumPool, GCNConv
        from spektral.transforms import  GCNFilter
        #transforms
        data_tr.apply(GCNFilter())
        data_te.apply(GCNFilter())
        
        #build model
        channels = data_tr.n_node_features
        num_labels = data_tr.n_labels
        class Net(Model):
            def __init__(self, channels, num_labels):
                super().__init__()
                self.conv1 = GCNConv(channels = channels//8)
                self.conv2 = GCNConv(channels = channels//16)
                self.conv3 = GCNConv(channels = channels//64)
                self.conv4 = GCNConv(channels = channels//128)
                self.conv5 = GCNConv(channels = channels//256)
                self.conv6 = GCNConv(channels = channels//512)
                self.pool = GlobalAttnSumPool()
                self.dense1 = Dense(512, activation = 'relu')
                self.dense2 = Dense(num_labels, activation="sigmoid")
                
            def call(self,inputs):
                x, a = inputs
                x = self.conv1([x,a])
                x = self.conv2([x,a])
                x = self.conv3([x,a])
                x = self.conv4([x,a])
                x = self.conv5([x,a])
                x = self.conv6([x,a])
                out = self.pool(x)
                out = self.dense1(out)
                out = self.dense2(out)
                
                return out
            
        #save path
        model_save_path = paths.get("GCN_save_path")
        
    elif model_name == "GCS":
        from spektral.layers import GlobalAttnSumPool, GCSConv
        from spektral.transforms import  GCNFilter
        #transforms
        data_tr.apply(GCNFilter())
        data_te.apply(GCNFilter())
        
        #build model
        channels = data_tr.n_node_features
        num_labels = data_tr.n_labels
        class Net(Model):
            def __init__(self, channels, num_labels):
                super().__init__()
                self.conv1 = GCSConv(channels = channels//8)
                self.conv2 = GCSConv(channels = channels//16)
                self.conv3 = GCSConv(channels = channels//64)
                self.conv4 = GCSConv(channels = channels//128)
                self.conv5 = GCSConv(channels = channels//256)
                self.conv6 = GCSConv(channels = channels//512)
                self.pool = GlobalAttnSumPool()
                self.dense1 = Dense(512, activation = 'relu')
                self.dense2 = Dense(num_labels, activation="sigmoid")
                
            def call(self,inputs):
                x, a = inputs
                x = self.conv1([x,a])
                x = self.conv2([x,a])
                x = self.conv3([x,a])
                x = self.conv4([x,a])
                x = self.conv5([x,a])
                x = self.conv6([x,a])
                out = self.pool(x)
                out = self.dense1(out)
                out = self.dense2(out)
                
                return out
            
        #save path
        model_save_path = paths.get("GCS_save_path")
        
    elif model_name == "GIN":
        from spektral.layers import GlobalAttnSumPool, GINConv
        from spektral.transforms import  LayerPreprocess
        from spektral.utils.sparse import sp_matrix_to_sp_tensor
        
        #transforms
        data_tr.apply(LayerPreprocess(GINConv))
        data_te.apply(LayerPreprocess(GINConv))
        data_tr.a = sp_matrix_to_sp_tensor(data_tr.a)
        data_te.a = sp_matrix_to_sp_tensor(data_te.a)
        
        #build model
        channels = data_tr.n_node_features
        num_labels = data_tr.n_labels
        class Net(Model):
            def __init__(self, channels, num_labels):
                super().__init__()
                self.conv1 = GINConv(channels=channels//2, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv2 = GINConv(channels=channels//4, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv3 = GINConv(channels = channels//8, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv4 = GINConv(channels = channels//16, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv5 = GINConv(channels = channels//32, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv6 = GINConv(channels = channels//64, mlp_hidden = [256, 256, 256, 256, 256])
                self.pool = GlobalAttnSumPool()
                # self.drop = Dropout(0.5)
                self.dense1 = Dense(512, activation = 'relu')
                self.dense2 = Dense(num_labels, activation="sigmoid")
                
            def call(self,inputs):
                x, a = inputs
                x = self.conv1([x,a])
                x = self.conv2([x,a])
                x = self.conv3([x,a])
                x = self.conv4([x,a])
                x = self.conv5([x,a])
                x = self.conv6([x,a])
                out = self.pool(x)
                # out = self.drop(out)
                out = self.dense1(out)
                out = self.dense2(out)
                
                return out
          
        #save path
        model_save_path = paths.get("GCS_save_path")  
        
        
    #Create model
    model = Net(channels, num_labels)
    
    #define training params
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr, epsilon = epsilon)
    loss_fn = BinaryCrossentropy()
    
    #packed batch mode loader
    loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_te = MixedLoader(data_te, batch_size=len(data_te))
    
    #define training function
    @tf.function
    def train_on_batch(inputs, target, sample_weights):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training = True)
            loss = loss_fn(target, predictions, sample_weight = sample_weights) +\
                sum(model.losses)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss
    
    #define test function    
    def test(loader):
        step = 0
        for batch in loader:
            step += 1
            inputs, targets = batch
            time_start = perf_counter()
            predictions = model(inputs,training = False)
            time_end = perf_counter()
            predictions = np.round_(predictions.numpy())
            targets = targets.reshape(len(targets),1)
            metrics = get_metrics(predictions, targets)
            test_inf_time = time_end - time_start
            if step == loader.steps_per_epoch:
                break

        return metrics, test_inf_time
            
    
    #train
    best_loss = 99999
    wait = 0
    step = 0
    epoch_loss = 0
    loss_tr = []
    if debug:
        epoch_counter = 0
    time_ref = perf_counter()
    for batch in loader_tr:
        
        step += 1
        inputs, targets = batch
        
        #get sample weights
        sample_weights = calc_sample_weights(targets, class_weights)
        
        #transpose targets
        target = np.reshape(targets,[len(targets),1])
        
        #train on batch
        loss = train_on_batch(inputs, target, sample_weights)
        epoch_loss += loss.numpy()
        
        #at the end of the epoch
        if step == loader_tr.steps_per_epoch:
            loss_tr.append(epoch_loss)
            best_loss, wait, break_from_loop = early_stop_calc(epoch_loss,
                                    1, best_loss, wait, patience)
            if break_from_loop:
                break
            epoch_loss = 0
            step = 0
            if debug:
                epoch_counter += 1
                print(f'epoch: {epoch_counter}\n')
            
    time_stop = perf_counter()
    train_time = time_stop - time_ref
    
    #test
    metrics, test_inf_time = test(loader_te)
    if isKfold == 0:
        model.save(model_save_path)
        
    return metrics, loss_tr, train_time, test_inf_time

def GNN_KFoldCrossVal(paths, params):

    #K-Fold cross validation
    #imports
    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.random import set_seed
    from numpy.random import seed
    from sklearn.model_selection import StratifiedKFold
    from tools import EEGDataset_GNN
    from tools import calc_class_weights
    from tools import get_label_list

    tf.config.list_physical_devices('GPU')
    seed(7)
    set_seed(7)
    
    #paths
    dataFilePath = paths.get('dataFilePath')
    results_path = paths.get('results_path')

    #parameters
    chanInd = params.get('chanInd')
    if chanInd is None:
        chanInd = list(range(19))
    k = params.get('k')   
    isKfold = 1;
    debug = params.get('debug')
    if debug is None:
        debug = False
    params['isKfold'] = isKfold
    one_hot = False
    is_graph = True
                
    #instantiate dataset
    IfMasterExists = 'GNN_Dataset' in locals()
    if IfMasterExists == False:
        GNN_Dataset = EEGDataset_GNN(dataFilePath, chanInd = chanInd, debug = 1)
        
    #determine class weight
    class_weights = calc_class_weights(GNN_Dataset, one_hot, is_graph)
    params['class_weights'] = class_weights

    #get list of labels for splits
    labels = get_label_list(GNN_Dataset, is_graph)
        
    #Kfold splits
    splits = StratifiedKFold(n_splits = k, shuffle= True, random_state=7)

    #define model names
    model_names = ["GCS", "GCN", "GIN"]
    #initialize metric values
    metrics_hist = np.zeros(shape = [10,k,len(model_names)])

    res = open(os.path.join(results_path, 'GNN_Results.txt'), 'w')

    #for every set in the list
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.zeros(len(GNN_Dataset)),labels)):
        
        if debug:
            print(f'fold: {fold}\n')

        #increment fold iter
        res.write("fold: {foldNum}\n".format(foldNum = fold))
        
        #create datasets
        #split data into training and testing
        data_tr = GNN_Dataset[train_idx]
        data_te = GNN_Dataset[val_idx]
        np.random.shuffle(data_tr)
        np.random.shuffle(data_te)
        data = {"data_tr":data_tr, "data_te":data_te}
        
        #train GNNs
        for model_ind in range(len(model_names)):
            
            if debug:
                print(f'model: {model_names[model_ind]}')

            #train on data, test on data, report results
            metrics, loss_hist, trainTime, test_inf_time = train_GNN(model_names[model_ind], data,
                                                 params, paths)
            res.write(model_names[model_ind] + ": " +\
                    "Acc = "+ str(metrics[0] * 100) +\
                      ", FPR = " + str(metrics[4] * 100) +\
                          ", TPR = " + str(metrics[3] * 100) +\
                              ", FNR = " + str(metrics[6] * 100) +\
                                  ", TNR = " + str(metrics[5] * 100) +\
                                      ", F1 = " + str(metrics[1]) +\
                                          ", epochs = " + str(len(loss_hist)) +\
                                              ", Train Time = " + str(trainTime) +\
                                                  ", Test Set Inference Time = " + str(test_inf_time) +'\n')
            metrics_hist[0:7,fold,model_ind] = metrics
            metrics_hist[7,fold,model_ind] = len(loss_hist)
            metrics_hist[8,fold,model_ind] = trainTime
            metrics_hist[9,fold,model_ind] = test_inf_time
        res.write('\n')
        
        
    #get mean and std
    metrics_mean = np.mean(metrics_hist, axis = 1)
    metrics_std = np.std(metrics_hist, axis = 1)

    #report results
    res.write('Means and Standard Deviations:\n')
    for model_ind in range(len(model_names)):
        res.write(model_names[model_ind] + " Mean: " +\
                  "Acc = "+ str(metrics_mean[0,model_ind] * 100) +\
                            ", FPR = " + str(metrics_mean[4,model_ind] * 100) +\
                                ", TPR = " + str(metrics_mean[3,model_ind] * 100) +\
                                    ", FNR = " + str(metrics_mean[6,model_ind] * 100) +\
                                        ", TNR = " + str(metrics_mean[5,model_ind] * 100) +\
                                            ", F1 = " + str(metrics_mean[1,model_ind]) +\
                                                ", epochs = " + str(metrics_mean[7,model_ind]) +\
                                                    ", Train Time = " + str(metrics_mean[8,model_ind]) +\
                                                        ", Test Set Inference Time = " + str(metrics_mean[9,model_ind]) +'\n')
        res.write(model_names[model_ind] + " Std: " +\
                  "Acc = "+ str(metrics_std[0,model_ind] * 100) +\
                            ", FPR = " + str(metrics_std[4,model_ind] * 100) +\
                                ", TPR = " + str(metrics_std[3,model_ind] * 100) +\
                                    ", FNR = " + str(metrics_std[6,model_ind] * 100) +\
                                        ", TNR = " + str(metrics_std[5,model_ind] * 100) +\
                                            ", F1 = " + str(metrics_std[1,model_ind]) +\
                                                ", epochs = " + str(metrics_std[7,model_ind]) +\
                                                    ", Train Time = " + str(metrics_std[8,model_ind]) +\
                                                        ", Test Set Inference Time = " + str(metrics_std[9,model_ind]) +'\n')
        
    res.close()
    
    return metrics_hist
    
    
def CNN_KFoldCrossVal(paths, params):
    
    import os
    import torch
    from torch import nn as nn
    from torch import device
    from torch.cuda import is_available
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from torch.optim import Adam
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from time import perf_counter
    from tools import EEGDataset_CNN_CSNN, calc_class_weights, get_label_list
    from tools import early_stop_calc, get_metrics
    
    #paths
    dataFilePath = paths.get('dataFilePath')
    results_path = paths.get('results_path')

    #parameters
    myDevice = device("cuda") if is_available() else device("cpu")
    k = params.get('k')
    batch_size = params.get('batch_size')
    num_epochs = params.get('num_epochs')
    patience = params.get('patience')
    chanInd = params.get('chanInd')
    if chanInd is None:
        chanInd = list(range(19))
    debug = params.get('debug')
    if debug is None:
        debug = False
    one_hot = True
    #optimizer parameters
    lr = 5e-4
    betas = (0.9, 0.999)
    
    torch.manual_seed(10)
                
    #instantiate EEG dataset
    dataset = EEGDataset_CNN_CSNN(dataFilePath, model_name = "CNN",
                                  chanInd = chanInd, debug = debug)

    #calculate class weight from dataset
    class_weights = calc_class_weights(dataset, one_hot)

    # training params
    weights = torch.from_numpy(np.array(class_weights))
    weights = weights.to(myDevice)
    loss_fn = nn.CrossEntropyLoss(weight = weights.float())
    
    #get list of labels
    labels = get_label_list(dataset)

    #Kfold splits
    splits = StratifiedKFold(n_splits = k, shuffle= True, random_state=7) 

    #initiate metrics list
    metrics_list = np.ones(shape = [k,10])

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.zeros(len(dataset)),
                                                            labels)):
        
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=len(val_idx),
                                 sampler=test_sampler)
        
        #initialize model
        net = nn.Sequential(nn.Conv2d(1, 12, 5),
                        nn.MaxPool2d(2),
                        nn.Conv2d(12, 64, 5),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(15744, 2),
                        nn.Sigmoid()
                        ).to(myDevice)
                        
        #optimizer
        optimizer = Adam(net.parameters(), lr=lr, betas=betas)
        #training
        smallest_loss = 999999
        wait = 0
        #start timer
        time_ref = perf_counter()
        # Outer training loop
        for epoch in range(num_epochs):    
            tot_loss = 0
            train_batch = iter(train_loader)
        
            # Minibatch training loop
            miniBatchIter = 0
            for data, targets in train_batch:
                miniBatchIter += 1
                
                #send data to device
                data = data.to(myDevice)
                targets = targets.to(myDevice)
        
                # forward pass
                net.train()
                out = net(data)
        
                # get loss
                loss_val = loss_fn(out, targets.float())
                tot_loss = tot_loss + float(loss_val)
        
                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
        
            #Early stopping
            smalles_loss, wait, break_from_loop = early_stop_calc(tot_loss,
                                miniBatchIter, smallest_loss, wait, patience)
            if break_from_loop:
                break
              
        #stop timer
        time_stop = perf_counter()
        train_time = time_stop - time_ref
        
        #evaluate
        with torch.no_grad():
            net.eval()
            for data, targets in iter(test_loader):
                data = data.to(myDevice)
                targets = targets.to(myDevice)
                
                # forward pass
                time_start = perf_counter()
                out = net(data)
                time_end = perf_counter()
                predicted = torch.round(out)
                test_inf_time = time_end - time_start
                
        #calculate metrics
        predicted_binary = predicted[:,1]
        targets_binary = targets[:,1]
        predicted_binary = predicted_binary.cpu()
        targets_binary = targets_binary.cpu()
        metrics = get_metrics(predicted_binary, targets_binary)
        
        #save results
        if fold+1 == 1:
            res = open(os.path.join(results_path, 'CNN_KFoldResults.txt'), 'w')
        else:
            res = open(os.path.join(results_path, 'CNN_KFoldResults.txt'), 'a')
        res.write("Fold: " + str(fold+1) + "\n")
        res.write("Acc = "+ str(metrics[0] * 100) +\
                  ", FPR = " + str(metrics[4] * 100) +\
                      ", TPR = " + str(metrics[3] * 100) +\
                          ", FNR = " + str(metrics[6] * 100) +\
                              ", TNR = " + str(metrics[5] * 100) +\
                                  ", F1 = " + str(metrics[1]) +\
                                      ", epochs = " + str(epoch) +\
                                          ", Train Time = " + str(train_time) +\
                                              ", Test Set Inference Time = " + str(test_inf_time) +'\n')
        res.close()
          
        metrics_list[fold,0:7] = metrics
        metrics_list[fold,7] = epoch
        metrics_list[fold,8] = train_time
        metrics_list[fold,9] = test_inf_time

    #calculate grand average
    grand_ave_metrics = np.mean(metrics_list, axis = 0)

    #calculate standard deviation
    grand_std_metrics = np.std(metrics_list, axis = 0)

    res = open(os.path.join(results_path, 'CNN_KFoldResults.txt'), 'a')
    res.write('\n')
    res.write("Grand Average:\n")
    res.write("Acc = "+ str(grand_ave_metrics[0] * 100) +\
              ", FPR = " + str(grand_ave_metrics[4] * 100) +\
                  ", TPR = " + str(grand_ave_metrics[3] * 100) +\
                      ", FNR = " + str(grand_ave_metrics[6] * 100) +\
                          ", TNR = " + str(grand_ave_metrics[5] * 100) +\
                              ", F1 = " + str(grand_ave_metrics[1]) +\
                                  ", epochs = " + str(grand_ave_metrics[7]) +\
                                      ", Train Time = " + str(grand_ave_metrics[8]) +\
                                          ", Test Set Inference Time = " + str(grand_ave_metrics[9]) +'\n')
    res.write("Standard Deviation:\n")
    res.write("Acc = "+ str(grand_std_metrics[0] * 100) +\
              ", FPR = " + str(grand_std_metrics[4] * 100) +\
                  ", TPR = " + str(grand_std_metrics[3] * 100) +\
                      ", FNR = " + str(grand_std_metrics[6] * 100) +\
                          ", TNR = " + str(grand_std_metrics[5] * 100) +\
                              ", F1 = " + str(grand_std_metrics[1]) +\
                                  ", epochs = " + str(grand_std_metrics[7]) +\
                                      ", Train Time = " + str(grand_std_metrics[8]) +\
                                          ", Test Set Inference Time = " + str(grand_std_metrics[9]) +'\n')
    res.close()
    
    return metrics_list


def CSNNSpikeConvert_KFoldCrossVal(paths, params):

    #same as train script 1 but adding conversion from regular data to spike data to note impact on training time
    import os
    import torch
    from torch import nn as nn
    from torch import device
    from torch.cuda import is_available
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from torch.optim import Adam
    import snntorch as snn
    from snntorch import surrogate
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from time import perf_counter
    from tools import EEGDataset_CNN_CSNN, calc_class_weights
    from tools import forward_pass, get_label_list, calc_loss, early_stop_calc
    from tools import get_metrics

    torch.manual_seed(7)
    
    #paths
    dataFilePath = paths.get('dataFilePath')
    results_path = paths.get('results_path')

    #parameters
    myDevice = device("cuda") if is_available() else device("cpu")
    batch_size = params.get('batch_size')
    num_epochs = params.get('num_epochs')
    patience = params.get('patience')
    k = params.get('k')
    chanInd = params.get('chanInd')
    if chanInd is None:
        chanInd = list(range(19))
    debug = params.get('debug')
    if debug is None:
        debug = False
    one_hot = True
    # Temporal Dynamics
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 25
    #optimizer parameters
    lr = 5e-4
    betas = (0.9, 0.999)
    

    #create loop that tries different threshold sizes and saves the results
    threshSizes = [0.625] #[0.1, 0.25, 0.5, 0.75, 1]; #[0.05, 0.1, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 1];
    metrics_hist = np.zeros(shape = [len(threshSizes), k, 9])

    for i in range(len(threshSizes)):
        res = open(os.path.join(results_path, 'CSNNSpikeConvert_KFoldResults.txt'),'a')
        thresh = threshSizes[i]
        res.write("Threshold: " + str(thresh) + '\n')
               
        #instantiate EEG dataset
        dataset = EEGDataset_CNN_CSNN(dataFilePath,
                        model_name = "CSNN_Spike_Convert", thresh = thresh,
                        chanInd = chanInd, debug = debug)
        
        #calculate class weight from 
        class_weights = calc_class_weights(dataset, one_hot)
        
        # training params
        weights = torch.from_numpy(np.array(class_weights))
        weights = weights.to(myDevice)
        loss_fn = nn.CrossEntropyLoss(weight = weights.float())
        dtype = torch.float
        
        #get labels
        labels = get_label_list(dataset)
        
        #Kfold splits
        splits = StratifiedKFold(n_splits = k, shuffle= True, random_state=7) 
        
        #initialize
        metrics_list = np.ones(shape = [k,9])
        
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.zeros(len(dataset)),
                                                                labels)):
        
            #define loaders
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size,
                                      sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=len(val_idx),
                                     sampler=test_sampler)
            
            #initialize model
            net = nn.Sequential(nn.Conv2d(1, 12, 5),
                            nn.MaxPool2d(2),
                            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                            nn.Conv2d(12, 64, 5),
                            nn.MaxPool2d(2),
                            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                            nn.Flatten(),
                            nn.Linear(15744, 2),
                            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                            ).to(myDevice)
            
            #optimizer
            optimizer = Adam(net.parameters(), lr=lr, betas=betas)
            
            #training
            smallest_loss = 999999
            wait = 0
            #start timer
            time_ref = perf_counter()
            # Outer training loop
            for epoch in range(num_epochs):
                tot_loss = 0
                train_batch = iter(train_loader)
            
                # Minibatch training loop
                miniBatchIter = 0
                for data, targets in train_batch:
                    miniBatchIter += 1
                    
                    #send data to device
                    data = data.to(myDevice)
                    targets = targets.to(myDevice)
            
                    # forward pass
                    net.train()
                    spk_rec, mem_rec = forward_pass(net, num_steps, data)
            
                    # initialize the loss & sum over time
                    loss_val = calc_loss(mem_rec, targets, num_steps, loss_fn,
                                         dtype, myDevice)
                    tot_loss = tot_loss + float(loss_val)
            
                    # Gradient calculation + weight update
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
            
                #Early stopping
                smallest_loss, wait, break_from_loop = early_stop_calc(tot_loss,
                    miniBatchIter, smallest_loss, wait, patience)         
                if break_from_loop:
                    break
                  
            #stop timer
            time_stop = perf_counter()
            train_time = time_stop - time_ref
            
            with torch.no_grad():
              net.eval()
              for data, targets in iter(test_loader):
                data = data.to(myDevice)
                targets = targets.to(myDevice)
                
                # forward pass
                time_start = perf_counter()
                test_spk, _ = forward_pass(net,num_steps,data)
                time_end = perf_counter()
                _, predicted = test_spk.sum(dim=0).max(1)
                test_inf_time = time_end - time_start
            
            #calculate metrics
            targets_for_metrics = targets[:,1]
            predicted = predicted.cpu()
            targets_for_metrics = targets_for_metrics.cpu()
            metrics = get_metrics(predicted, targets_for_metrics)
            
            #save results
            res.write("Fold: " + str(fold+1) + "\n")
            res.write("Acc = "+ str(metrics[0] * 100) +\
                      ", FPR = " + str(metrics[4] * 100) +\
                          ", TPR = " + str(metrics[3] * 100) +\
                              ", FNR = " + str(metrics[6] * 100) +\
                                  ", TNR = " + str(metrics[5] * 100) +\
                                      ", F1 = " + str(metrics[1]) +\
                                          ", epochs = " + str(epoch) +\
                                              ", Train Time = " + str(train_time) +\
                                                  ", Test Set Inference Time = " + str(test_inf_time) +'\n')
              
            metrics_list[fold,0:7] = metrics
            metrics_list[fold,7] = epoch
            metrics_list[fold,8] = train_time
            metrics_list[fold,9] = test_inf_time
        
        #calculate grand average
        grand_ave_metrics = np.mean(metrics_list, axis = 0)

        #calculate standard deviation
        grand_std_metrics = np.std(metrics_list, axis = 0)
        
        res.write('\n')
        res.write("Grand Average:\n")
        res.write("Acc = "+ str(grand_ave_metrics[0] * 100) +\
                  ", FPR = " + str(grand_ave_metrics[4] * 100) +\
                      ", TPR = " + str(grand_ave_metrics[3] * 100) +\
                          ", FNR = " + str(grand_ave_metrics[6] * 100) +\
                              ", TNR = " + str(grand_ave_metrics[5] * 100) +\
                                  ", F1 = " + str(grand_ave_metrics[1]) +\
                                      ", epochs = " + str(grand_ave_metrics[7]) +\
                                          ", Train Time = " + str(grand_ave_metrics[8]) +\
                                              ", Test Set Inference Time = " + str(grand_ave_metrics[9]) +'\n')
        res.write("Standard Deviation:\n")
        res.write("Acc = "+ str(grand_std_metrics[0] * 100) +\
                  ", FPR = " + str(grand_std_metrics[4] * 100) +\
                      ", TPR = " + str(grand_std_metrics[3] * 100) +\
                          ", FNR = " + str(grand_std_metrics[6] * 100) +\
                              ", TNR = " + str(grand_std_metrics[5] * 100) +\
                                  ", F1 = " + str(grand_std_metrics[1]) +\
                                      ", epochs = " + str(grand_std_metrics[7]) +\
                                          ", Time = " + str(grand_std_metrics[8]) +\
                                              ", Test Set Inference Time = " + str(grand_std_metrics[9]) +'\n')
        res.write('\n\n')
        
        res.close()
        
        #append to metrics_hist 
        metrics_hist[i,:,:] = metrics_list
    
    return metrics_hist


def EEGNet_KFoldCrossVal(paths, params):
    
    #define imports
    import tensorflow as tf
    import numpy as np
    from os.path import join
    from time import perf_counter
    from sklearn.model_selection import StratifiedKFold
    from EEGModels import EEGNet
    from tools import Load_exp1_data_EEGNet, calc_class_weights_EEGNet
    from tools import get_labels_EEGNet, calc_sample_weights_EEGNet
    from tools import get_split_dataset_EEGNet, early_stop_calc, get_metrics
    
    np.random.seed(7)
    tf.random.set_seed(7)
    
    #set random seed
    tf.random.set_seed(7)
    
    #grab paths
    dataFilePath = paths.get('dataFilePath')
    results_path = paths.get('results_path')
    
    #define parameters
    k = params.get('k')
    batch_size = params.get('batch_size')
    num_epochs = params.get('num_epochs')
    patience = params.get('patience')
    debug = params.get('debug')
    if debug is None:
        debug = False
    chanInd = params.get('chanInd')
    if chanInd is None:
        chanInd = list(range(19))
    opt_lr = 5e-5
    betas = (0.9, 0.999)
    one_hot = True
        
    #instantiate dataset
    dataset, dataset_length = Load_exp1_data_EEGNet(dataFilePath, one_hot, chanInd)
    
    #calculate class weights
    class_weights = calc_class_weights_EEGNet(dataset, one_hot)
    
    #get list of labels
    labels = get_labels_EEGNet(dataset, one_hot)
    
    #kfold splits
    splits = StratifiedKFold(n_splits = k, shuffle = True, random_state = 7)
    
    #initiate metrics list
    metrics_list = np.ones(shape = [k,10])
    
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.zeros(dataset_length),
                                                             labels)):
        
        #initialize optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = opt_lr,
                                             beta_1 = betas[0], beta_2 = betas[1])
        
        #define loss
        loss_fcn = tf.keras.losses.CategoricalCrossentropy()
        
        #get training and testing data sets
        train_data = get_split_dataset_EEGNet(dataset, train_idx)
        train_data = train_data.batch(batch_size)
        test_data = get_split_dataset_EEGNet(dataset, val_idx)
        test_data = test_data.batch(len(val_idx))
        
        #configure model with defaults from example
        model = EEGNet(nb_classes = 2, Chans = 19, Samples = 996)
        
        #training
        smallest_loss = 1e10
        wait = 0
        #start timer
        time_ref = perf_counter()
        for ep_cnt in range(num_epochs):
            ep_loss = 0
            miniBatchIter = 0
            for step, (inputs, targets) in enumerate(train_data):
                miniBatchIter += 1
                with tf.GradientTape() as tape:
                    #forward pass
                    predictions = model(inputs)
                    
                    #calculate sample weights
                    sample_weights = calc_sample_weights_EEGNet(targets, class_weights,
                                                         one_hot)
                    
                    # get loss
                    loss = loss_fcn(targets, predictions,
                                    sample_weight = tf.reshape(sample_weights,
                                                               shape = [len(sample_weights),1]))
                    ep_loss += loss.numpy()
                    
                #compute gradients
                grads = tape.gradient(loss, model.trainable_variables)
                
                #apply gradients
                optimizer.apply_gradients([(g,v) for g,v in zip(grads, 
                                                                model.trainable_variables)])
                
            #check early stopping
            smallest_loss, wait, break_from_loop = early_stop_calc(ep_loss,
                        miniBatchIter, smallest_loss, wait, patience)
            if break_from_loop:
                break
                
        #stop timer
        time_stop = perf_counter()
        train_time = time_stop - time_ref
        
        #evaluate
        for step, (inputs, targets) in enumerate(test_data):
            #get predictions
            time_ref = perf_counter()
            predictions = model(inputs)
            time_stop = perf_counter()
            if step == 0:
                total_predictions = predictions
                total_targets = targets
                test_inf_time = time_stop - time_ref
            else:
                total_predictions = tf.concat([total_predictions, predictions], axis = 0)
                total_targets = tf.concat([total_targets, targets], axis = 0)
                test_inf_time += time_stop - time_ref   
            
        #get metrics
        predictions = tf.round(predictions[:,1])
        targets = targets[:,1]
        metrics = get_metrics(predictions, targets)
        
        #save results
        if fold+1 == 1:
            res = open(join(results_path, "EEGNet_KFoldResults.txt"), 'w')
        else:
            res = open(join(results_path, "EEGNet_KFoldResults.txt"), 'a')
        res.write("Fold: " + str(fold+1) + "\n")
        res.write("Acc = "+ str(metrics[0] * 100) +\
                  ", FPR = " + str(metrics[4] * 100) +\
                      ", TPR = " + str(metrics[3] * 100) +\
                          ", FNR = " + str(metrics[6] * 100) +\
                              ", TNR = " + str(metrics[5] * 100) +\
                                  ", F1 = " + str(metrics[1]) +\
                                      ", epochs = " + str(ep_cnt) +\
                                          ", Train Time = " + str(train_time) +\
                                              ", Test Set Inference Time = " + str(test_inf_time) + '\n')
        res.close()
        
        metrics_list[fold,0:7] = metrics
        metrics_list[fold, 7] = ep_cnt
        metrics_list[fold, 8] = train_time
        metrics_list[fold, 9] = test_inf_time
        
    #calculate grand average and std
    grand_ave_metrics = np.mean(metrics_list, axis = 0)
    grand_std_metrics = np.std(metrics_list, axis = 0)
    
    #save
    res = open(join(results_path, 'EEGNet_KFoldResults.txt'), 'a')
    res.write('\n')
    res.write("Grand Average:\n")
    res.write("Acc = "+ str(grand_ave_metrics[0] * 100) +\
              ", FPR = " + str(grand_ave_metrics[4] * 100) +\
                  ", TPR = " + str(grand_ave_metrics[3] * 100) +\
                      ", FNR = " + str(grand_ave_metrics[6] * 100) +\
                          ", TNR = " + str(grand_ave_metrics[5] * 100) +\
                              ", F1 = " + str(grand_ave_metrics[1]) +\
                                  ", epochs = " + str(grand_ave_metrics[7]) +\
                                      ", Train Time = " + str(grand_ave_metrics[8]) +\
                                          ", Test Set Inference Time = " + str(grand_ave_metrics[9]) + '\n')
    res.write("Standard Deviation:\n")
    res.write("Acc = "+ str(grand_std_metrics[0] * 100) +\
              ", FPR = " + str(grand_std_metrics[4] * 100) +\
                  ", TPR = " + str(grand_std_metrics[3] * 100) +\
                      ", FNR = " + str(grand_std_metrics[6] * 100) +\
                          ", TNR = " + str(grand_std_metrics[5] * 100) +\
                              ", F1 = " + str(grand_std_metrics[1]) +\
                                  ", epochs = " + str(grand_std_metrics[7]) +\
                                      ", Train Time = " + str(grand_std_metrics[8]) +\
                                          ", Test Set Inference Time = " + str(grand_std_metrics[9]) + '\n')
    res.close()
    
    return metrics_list