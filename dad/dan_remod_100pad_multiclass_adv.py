
# coding: utf-8

# In[27]:


# import csv files
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tqdm import tqdm
import utils
from torchnet.meter import ConfusionMeter
#from torch.utils.tensorboard import SummaryWriter

import logging
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
filename = timestr + '.log'
logging.basicConfig(level=logging.INFO, filename=filename)
logging.getLogger().setLevel(logging.INFO)

from torch.autograd import Function
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic=True

class best_eval():
    def __init__(self):
        self.best_dev_score = 0
        self.current_dev_score = 0

def proc_data_prep():
    #956 back_train.csv
    #19 land_train.csv
    #41214 neptune_train.csv
    #67343 normal_train.csv
    #201 pod_train.csv
    #2646 smurf_train.csv
    #892 teardrop_train.csv
    #45928 malicious

    train_normal_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/train_normal.csv',header=None)
    train_dos_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/train_dos.csv',header=None)
    train_probe_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/train_probe.csv',header=None)
    train_u2r_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/train_u2r.csv',header=None)
    train_r2l_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/train_r2l.csv',header=None)

    dev_normal_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/dev_normal.csv',header=None)
    dev_dos_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/dev_dos.csv',header=None)
    dev_probe_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/dev_probe.csv',header=None)
    dev_u2r_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/dev_u2r.csv',header=None)
    dev_r2l_data = pd.read_csv('/mnt/matylda6/project_evo/dan/kddcup99-cnn/processed_data/dev_r2l.csv',header=None)

    #train_src_data = pd.concat([train_normal_data, train_dos_data, train_r2l_data, train_probe_data])
    train_src_data = pd.concat([train_normal_data, train_dos_data, train_probe_data])
    #dev_src_data = pd.concat([dev_normal_data, dev_dos_data, dev_r2l_data, dev_probe_data])
    dev_src_data = pd.concat([dev_normal_data, dev_dos_data, dev_probe_data])
    train_tgt_data = pd.concat([dev_u2r_data])
    dev_tgt_data = pd.concat([dev_u2r_data])

    train_src_x = torch.Tensor(train_src_data.iloc[:,:-2].astype(np.float32).to_numpy())#[:,5:]
    train_tgt_x = torch.Tensor(train_tgt_data.iloc[:,:-2].astype(np.float32).to_numpy())#[:,5:]
    dev_src_x = torch.Tensor(dev_src_data.iloc[:,:-2].astype(np.float32).to_numpy())#[:,5:]
    dev_tgt_x = torch.Tensor(dev_tgt_data.iloc[:,:-2].astype(np.float32).to_numpy())#[:,5:]

    train_str_labels = train_src_data.iloc[:, -2].to_numpy()
    labels_a = np.where(train_str_labels=='normal', 0, train_str_labels)
    labels_b = np.where(labels_a=='malicious', 1, labels_a)
    train_src_labels = labels_b.astype(int)
    # dos-1,u2r-2,r2l-3,probe-4
    train_str_labels = train_tgt_data.iloc[:, -2].to_numpy()
    labels_a = np.where(train_str_labels=='normal', 0, train_str_labels)
    labels_b = np.where(labels_a=='malicious', 1, labels_a)
    train_tgt_labels = labels_b.astype(int)
 
    dev_str_labels = dev_src_data.iloc[:, -2].to_numpy()
    labels_a = np.where(dev_str_labels=='normal', 0, dev_str_labels)
    labels_b = np.where(labels_a=='malicious', 1, labels_a)
    dev_src_labels = labels_b.astype(int)
    dev_str_labels = dev_tgt_data.iloc[:, -2].to_numpy()
    labels_a = np.where(dev_str_labels=='normal', 0, dev_str_labels)
    labels_b = np.where(labels_a=='malicious', 1, labels_a)
    dev_tgt_labels = labels_b.astype(int)
 
    train_src_y = torch.Tensor(train_src_labels.astype(np.int64))
    train_tgt_y = torch.Tensor(train_tgt_labels.astype(np.int64))
    dev_src_y =  torch.Tensor(dev_src_labels.astype(np.int64))
    dev_tgt_y =  torch.Tensor(dev_tgt_labels.astype(np.int64))
    
    return train_src_x, train_tgt_x, dev_src_x, dev_tgt_x, train_src_y, train_tgt_y, dev_src_y, dev_tgt_y

def data_prep():
    #956 back_train.csv
    #19 land_train.csv
    #41214 neptune_train.csv
    #67343 normal_train.csv
    #201 pod_train.csv
    #2646 smurf_train.csv
    #892 teardrop_train.csv
    #45928 malicious

    train_data = pd.read_csv('/mnt/matylda6/project_evo/dan/datasets/KDD/KDDTrain_4l+.txt',header=None)
    #train_data = pd.read_csv('/mnt/matylda6/project_evo/dan/datasets/KDD/KDDTrain_2l.csv',header=None)
    tlen = len(train_data)
    #dev_data = pd.read_csv('/mnt/matylda6/project_evo/dan/datasets/KDD/KDDTest_2l.csv',header=None)
    #dev_data = pd.read_csv('/mnt/matylda6/project_evo/dan/datasets/KDD/KDDTest_4l_filt+.txt',header=None)
    dev_data = pd.read_csv('/mnt/matylda6/project_evo/dan/datasets/KDD/KDDTest_4l+.txt',header=None)
    dlen = len(dev_data)

    data = pd.concat([train_data, dev_data])

    #data = pd.get_dummies(data.iloc[:,:-2])

    proto = LabelEncoder()
    proto.fit(data.iloc[:,1].to_numpy())
    data.iloc[:,1] = proto.transform(data.iloc[:,1].to_numpy())
    service = LabelEncoder()
    service.fit(data.iloc[:,2].to_numpy())
    data.iloc[:,2] = service.transform(data.iloc[:,2].to_numpy())
    flag = LabelEncoder()
    flag.fit(data.iloc[:,3].to_numpy())
    data.iloc[:,3] = flag.transform(data.iloc[:,3].to_numpy())

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data.iloc[:,:-2].to_numpy())
    new_data = scaler.transform(data.iloc[:,:-2].to_numpy()) 
    #new_data = data.to_numpy()
    #train_all_data = data.iloc[:tlen,:-2]
    train_all_data = new_data[:tlen]
    dev_all_data = new_data[tlen:tlen+dlen]
    #train_all_data = np.pad(train_all_data.to_numpy(), ((0, 0), (0, 64 - len(train_all_data.to_numpy()))), 'constant').reshape(-1, 1, 8, 8)
    #dev_all_data = data.iloc[tlen:tlen+dlen,:-2]
    #dev_all_data = np.pad(dev_all_data.to_numpy(), ((0, 0), (0, 64 - len(dev_all_data.to_numpy()))), 'constant').reshape(-1, 1, 8, 8)
    #dev_all_data = pd.get_dummies(dev_data.iloc[:,:-2])
    
    np.savetxt("train_x.csv", train_all_data.astype(np.float32), delimiter=",")
    np.savetxt("dev_x.csv", dev_all_data.astype(np.float32), delimiter=",")
    #train_x = torch.Tensor(train_all_data.to_numpy().astype(np.float32))#[:,5:]
    train_x = torch.Tensor(train_all_data.astype(np.float32))#[:,5:]
    #dev_x =  torch.Tensor(dev_all_data.to_numpy().astype(np.float32))#[:,5:]
    dev_x =  torch.Tensor(dev_all_data.astype(np.float32))#[:,5:]
    mean_x = torch.mean(train_x)
    std_x = torch.std(train_x)
    #train_x = (train_x - mean_x) / std_x
    #dev_x = (dev_x - mean_x) / std_x
    test_x = dev_x

    train_str_labels = train_data.iloc[:, -2].to_numpy()
    labels_a = np.where(train_str_labels=='normal', 0, train_str_labels)
    labels_b = np.where(labels_a=='DOS', 1, labels_a)
    labels_c = np.where(labels_b=='U2R', 1, labels_b)
    labels_d = np.where(labels_c=='R2L', 1, labels_c)
    labels_e = np.where(labels_d=='PROBE', 1, labels_d)
    train_labels = labels_e.astype(int)
    # dos-1,u2r-2,r2l-3,probe-4
    labels_a = np.where(train_str_labels=='normal', 0, train_str_labels)
    labels_b = np.where(labels_a=='DOS', 1, labels_a)
    labels_c = np.where(labels_b=='U2R', 2, labels_b)
    labels_d = np.where(labels_c=='R2L', 3, labels_c)
    labels_e = np.where(labels_d=='PROBE', 4, labels_d)
    train_ds_labels = labels_e.astype(int)

    dev_str_labels = dev_data.iloc[:, -2].to_numpy()
    labels_a = np.where(dev_str_labels=='normal', 0, dev_str_labels)
    labels_b = np.where(labels_a=='DOS', 1, labels_a)
    labels_c = np.where(labels_b=='U2R', 1, labels_b)
    labels_d = np.where(labels_c=='R2L', 1, labels_c)
    labels_e = np.where(labels_d=='PROBE', 1, labels_d)
    dev_labels = labels_e.astype(int)
    # dos-1,u2r-2,r2l-3,probe-4
    labels_b = np.where(labels_a=='DOS', 1, labels_a)
    labels_c = np.where(labels_b=='U2R', 2, labels_b)
    labels_d = np.where(labels_c=='R2L', 3, labels_c)
    labels_e = np.where(labels_d=='PROBE', 4, labels_d)
    dev_ds_labels = labels_e.astype(int)

    np.savetxt("train_2class.csv", train_labels.astype(np.int64), delimiter=",")
    np.savetxt("train_5class.csv", train_ds_labels.astype(np.int64), delimiter=",")
    np.savetxt("dev_2class.csv", dev_labels.astype(np.int64), delimiter=",")
    np.savetxt("dev_5class.csv", dev_ds_labels.astype(np.int64), delimiter=",")
    train_y = torch.Tensor(train_labels.astype(np.int64))
    train_z = torch.Tensor(train_ds_labels.astype(np.int64))
    dev_y =  torch.Tensor(dev_labels.astype(np.int64))
    dev_z =  torch.Tensor(dev_ds_labels.astype(np.int64))
    test_y =  dev_y
    test_z =  dev_z
    
    return train_x, dev_x, test_x, train_y, dev_y, test_y, train_z, dev_z, test_z


def save_best_model(model, save_dir, model_name):
    """
    :param model:  nn model
    :param save_dir:  save model direction
    :param model_name:  model name
    :param best_eval:  eval best
    :return:  None
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = "{}.pt".format(model_name)
    save_path = os.path.join(save_dir, model_name)
    print("save best model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close()
    best_eval.early_current_patience = 0

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dr=0.0):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dr, bidirectional = True)
        self.lstm_b = nn.LSTM(hidden_dim*2, hidden_dim, num_layers, batch_first=True, dropout=dr, bidirectional = True)
        # Readout layer
        self.fcb = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(device)
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, (hn,cn) = self.lstm(x, (h0,c0))
        #out, (hm,cm) = self.lstm_b(out, (hn,cn))

        out = self.fcb(out[:, :, :])
        return out

class AttackTypeClassifier(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size, dropout, batch_norm=False):
        super(AttackTypeClassifier, self).__init__()
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())
            #self.net.add_module('p-sigmoid-{}'.format(i), nn.Sigmoid())
            self.net.add_module('p-linear-final', nn.Linear(hidden_dim, output_size))
            #self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))
    def forward(self, inp):
        return self.net(inp)

class AttackTypeDiscriminator(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size, dropout, batch_norm=False):
        super(AttackTypeDiscriminator, self).__init__()
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())
            self.net.add_module('q-linear-final', nn.Linear(hidden_dim, output_size))
            #self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))
    def forward(self, inp):
        return self.net(inp)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size, kernel_num, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, kernel_size=(K, input_dim)) for K in kernel_sizes])
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(
                    len(kernel_sizes)*kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())
            #self.fcnet.add_module('f-sigmoid-{}'.format(i), nn.Sigmoid())
    def forward(self, inp):
        out = [torch.relu(conv(inp)).squeeze(3) for conv in self.convs]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1) 
        return self.fcnet(out)

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dropout, bdrnn=True):
        super(LSTMFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim//2 if bdrnn else hidden_dim
        self.n_cells = self.num_layers*2 if bdrnn else num_layers
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bdrnn)
        self.fcb = nn.Linear(hidden_dim, hidden_dim) 

    def forward(self, inp):
        inp = inp.view(1, -1, input_dim)
        h0 = torch.zeros(self.n_cells, inp.size(1), self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_cells, inp.size(1), self.hidden_dim).to(device)
        output, (ht, ct) = self.rnn(inp, (h0, c0))
        output = self.fcb(output[:, :, :]) 

        return output[0]

save_dir='results_kdd_multiclass_cnn2d'
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev) 

train_src_x, train_tgt_x, dev_src_x, dev_tgt_x, train_src_y, train_tgt_y, dev_src_y, dev_tgt_y = proc_data_prep()
input_dim = train_src_x.size(1)
hidden_dim = 1024
layer_dim = 1
output_dim = 2
max_epochs = 20
learning_rate = 0.0005
seq_len = 1
patience = 5
# Parameters
batch_size = 1024
params = {'batch_size': batch_size,
          'shuffle': True,
          'pin_memory': True,
          'num_workers': 8,
          'worker_init_fn': np.random.seed(seed)}
val_bs=1024
valid_params = {'batch_size': val_bs, 'shuffle': False}


wait_time = 0
num_iter = 0
mtype='adan'
if mtype == 'cnn' or mtype == 'dann2d':
    # do padding
    import torch.nn.functional as F
    train_x = F.pad(input=train_x, pad=(0, 121 - input_dim), mode='constant').reshape(-1, 1, 11, 11)
    dev_x = F.pad(input=dev_x, pad=(0, 121 - input_dim), mode='constant').reshape(-1, 1, 11, 11)

if mtype == 'adan':
    import torch.nn.functional as F
    train_src_x = train_src_x.unsqueeze(1).unsqueeze(2)
    train_tgt_x = train_tgt_x.unsqueeze(1).unsqueeze(2)
    dev_src_x = dev_src_x.unsqueeze(1).unsqueeze(2)
    dev_tgt_x = dev_tgt_x.unsqueeze(1).unsqueeze(2)
    train_src_y = train_src_y
    dev_src_y = dev_src_y
    train_tgt_y = train_tgt_y
    dev_tgt_y = dev_tgt_y


# Datasets
training_src_set = TensorDataset(train_src_x, train_src_y)
training_tgt_set = TensorDataset(train_tgt_x, train_tgt_y)
validation_src_set = TensorDataset(dev_src_x, dev_src_y)
validation_tgt_set = TensorDataset(dev_tgt_x, dev_tgt_y)

training_src_generator = DataLoader(training_src_set, **params)
training_src_generator_Q = DataLoader(training_src_set, **params)
training_tgt_generator = DataLoader(training_tgt_set, **params)
training_tgt_generator_Q = DataLoader(training_tgt_set, **params)

train_src_iter_Q = iter(training_src_generator_Q)
train_tgt_iter = iter(training_tgt_generator)
train_tgt_iter_Q = iter(training_tgt_generator_Q)

validation_src_generator = DataLoader(validation_src_set, **valid_params)
validation_tgt_generator = DataLoader(validation_tgt_set, **valid_params)

num_layers=5
FE = CNNFeatureExtractor(input_dim, 1, hidden_dim, 40, [1,1,1,1,1,1,1,1], 0.0)
#FE = LSTMFeatureExtractor(input_dim, num_layers, hidden_dim, 0.5, bdrnn=True)
CL = AttackTypeClassifier(1, hidden_dim, 2, 0.5, True)
DS = AttackTypeDiscriminator(1, hidden_dim, 2, 0.5, True)

FE, CL, DS = FE.to(device), CL.to(device), DS.to(device)

optimizer = torch.optim.Adam(list(FE.parameters()) + list(CL.parameters()), lr=learning_rate)  
optimizerQ = torch.optim.Adam(DS.parameters(), lr=learning_rate)  
def train(train_src_iter_Q, train_tgt_iter_Q,train_tgt_iter):
    model_save_file='models_trained'
    criterion = nn.CrossEntropyLoss()
    xe_ds = nn.CrossEntropyLoss()
    beta = 0.0
    n_critic = 5
    best_acc = 0.0
    for epoch in range(max_epochs):
        print('epoch {}'.format(epoch + 1))
        FE.train()
        CL.train()
        DS.train()
        train_src_iter = iter(training_src_generator)
        # training accuracy
        correct, total = 0, 0
        for i, (inputs_src, targets_src) in tqdm(enumerate(train_src_iter),
                total=len(training_src_set)//batch_size):
            targets_src = targets_src.long().to(device)
            try:
                inputs_tgt, labels_tgt = next(train_tgt_iter)  # target (U2R) labels are not used
            except:
                # check if Chinese data is exhausted
                train_tgt_iter = iter(training_tgt_generator)
                inputs_tgt, labels_tgt = next(train_tgt_iter)

            # Q iterations
            if n_critic>0 and ((epoch==0 and i<=25) or (i%500==0)):
                n_critic = 5
            utils.freeze_net(FE)
            utils.freeze_net(CL)
            utils.unfreeze_net(DS)
            for qiter in range(n_critic):
                DS.zero_grad()
                # get a minibatch of data
                try:
                    # labels are not used
                    q_inputs_src, _ = next(train_src_iter_Q)
                except StopIteration:
                    # check if dataloader is exhausted
                    train_src_iter_Q = iter(training_src_generator_Q)
                    q_inputs_src, _ = next(train_src_iter_Q)
                try:
                    q_inputs_tgt, _ = next(train_tgt_iter_Q)
                except StopIteration:
                    train_tgt_iter_Q = iter(training_tgt_generator_Q)
                    q_inputs_tgt, _ = next(train_tgt_iter_Q)

                features_src = FE(q_inputs_src.to(device))
                o_src_ad = DS(features_src)
                l_src_ad = torch.mean(o_src_ad)
                (-l_src_ad).backward()
                #logging.debug(f'DS grad norm: {DS.net[1].weight.grad.data.norm()}')
                
                features_tgt = FE(q_inputs_tgt.to(device))
                o_tgt_ad = DS(features_tgt)
                l_tgt_ad = torch.mean(o_tgt_ad)
                (l_tgt_ad).backward()
                #logging.debug(f'DS grad norm: {DS.net[1].weight.grad.data.norm()}')
                
                optimizerQ.step()

            #logging.info('Src ds loss {}'.format(l_src_ad))
            #logging.info('Tgt ds loss {}'.format(l_tgt_ad))
 
            # FE&CL iteration
            utils.unfreeze_net(FE)
            utils.unfreeze_net(CL)
            utils.freeze_net(DS)
            ### clip Q weights
            for p in DS.parameters():
                p.data.clamp_(0.01, 0.01)
            FE.zero_grad()
            CL.zero_grad()
          
            features_src = FE(inputs_src.to(device))
            features_tgt = FE(inputs_tgt.to(device))
            labels_tgt = labels_tgt.long().to(device)

            o_src_sent = CL(features_src)
            l_src_sent = criterion(o_src_sent, targets_src)
            #o_tgt_sent = CL(features_tgt)
            #l_tgt_sent = criterion(o_tgt_sent, labels_tgt)
            #l_src_sent += l_tgt_sent 
            ##print('Loss src {}'.format(l_src_sent))
            l_src_sent.backward(retain_graph=True)
            #o_src_ad = DS(features_src)
            #l_src_ad = torch.mean(o_src_ad)
            #(-beta*l_src_ad).backward(retain_graph=True)

            #
            # training accuracy
            _, pred = torch.max(o_src_sent, 1)
            total += targets_src.size(0)
            correct += (pred == targets_src).sum().item()
            
            features_tgt = FE(inputs_tgt.to(device))
            o_tgt_ad = DS(features_tgt)
            l_tgt_ad = torch.mean(o_tgt_ad)
            (beta*l_tgt_ad).backward()
            
            optimizer.step()
        
        # end of epoch
        logging.info('Src FE and CL, ds loss {}'.format(l_src_ad))
        logging.info('Tgt FE and CL, ds loss {}'.format(l_tgt_ad))
        logging.info('Ending epoch {}'.format(epoch+1))
        # logs
        #    logging.info(f'Average Tgt Q output: {sum_tgt_q[1]/sum_tgt_q[0]}')
        # evaluate
        logging.info('Training Accuracy: {}%'.format(100.0*correct/total))
        #print('Training Accuracy: {}%'.format(100.0*correct/total))
        logging.info('Evaluating Source Validation set:')
        evaluate(validation_src_generator, FE, CL)
        logging.info('Evaluating Target validation set:')
        acc = evaluate(validation_tgt_generator, FE, CL)
        if acc > best_acc:
            logging.info(f'New Best target validation accuracy: {acc}')
            best_acc = acc
            torch.save(FE.state_dict(),
                    '{}/netF_epoch_{}.pth'.format(model_save_file, epoch))
            torch.save(CL.state_dict(),
                    '{}/netP_epoch_{}.pth'.format(model_save_file, epoch))
            torch.save(DS.state_dict(),
                    '{}/netQ_epoch_{}.pth'.format(model_save_file, epoch))
    logging.info(f'Best target validation accuracy: {best_acc}')


def evaluate(loader, FE, CL):
    FE.eval()
    CL.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(2)
    with torch.no_grad():
        for inputs, targets in tqdm(it):
            targets = targets.long().to(device)
            outputs = CL(FE(inputs.to(device)))
            _, pred = torch.max(outputs, 1)
            confusion.add(pred.data, targets.data)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
    accuracy = correct / total
    logging.info('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
    #print('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
    logging.info(confusion.conf)
    return accuracy


def test():
        model.load_state_dict(torch.load(save_dir +'/model.best.pt'))
        eval_norm = 0
        eval_mal = 0
        eval_dos = 0
        eval_u2r = 0
        eval_r2l = 0
        eval_prob = 0
        eval_norm_dict = {'support': 0, 'precision': 0, 'f1-score': 0, 'recall':0}
        eval_mal_dict = {'support': 0, 'precision': 0, 'f1-score': 0, 'recall':0}
        eval_acc = 0
        for data in validation_generator:
            if output_dim == 2:
                img, label, _ = data
            elif output_dim == 5:
                img, _, label = data
            else:
                break
            if mtype == 'cnn1d' or mtype == 'cnnb':
                img = img.view(-1, 1, input_dim).to(device)
            elif mtype == 'lstm':
                img = img.view(-1, 1, input_dim).to(device)


            if USE_GPU:
                img = img.cuda()
                label = label.long().cuda()
            out = model(img)
            if mtype == 'lstm':
                out=out.squeeze(1)

            _, pred = torch.max(out, 1)
            if output_dim == 2:
                target_names = ['normal', 'malicious']
                matrix = sklearn.metrics.multilabel_confusion_matrix(label.long().cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=[0,1])
                out_dict = sklearn.metrics.classification_report(label.long().cpu().detach().numpy(), pred.cpu().detach().numpy(), target_names=target_names, labels=[0,1], output_dict=True)
                norm_dict = out_dict['normal']
                mal_dict = out_dict['malicious']
                eval_norm_dict = Counter(eval_norm_dict) + Counter(norm_dict)
                eval_mal_dict = Counter(eval_mal_dict) + Counter(mal_dict)
                normal_acc = matrix[0,1][1]/ (matrix[0,0][1] + matrix[0,1][1])
                malicious_acc = matrix[1,1][1]/ (matrix[1,0][1] + matrix[1,1][1])
                eval_norm += normal_acc
                eval_mal += malicious_acc
            elif output_dim == 5:
                target_names = ['normal', 'DOS', 'U2R', 'R2L', 'PROBE']
                matrix = sklearn.metrics.multilabel_confusion_matrix(label.long().cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=[0,1,2,3,4])
                out_dict = sklearn.metrics.classification_report(label.long().cpu().detach().numpy(), pred.cpu().detach().numpy(), target_names=target_names, labels=[0,1,2,3,4], output_dict=True)
                normal_acc = matrix[0,1][1]/ (matrix[0,0][1] + matrix[0,1][1])
                dos_acc = matrix[1,1][1]/ (matrix[1,0][1] + matrix[1,1][1])
                u2r_acc = matrix[2,1][1]/ (matrix[2,0][1] + matrix[2,1][1])
                r2l_acc = matrix[3,1][1]/ (matrix[3,0][1] + matrix[3,1][1])
                prob_acc = matrix[4,1][1]/ (matrix[4,0][1] + matrix[4,1][1])
                eval_norm += normal_acc
                eval_dos += dos_acc
                eval_u2r += u2r_acc
                eval_r2l += r2l_acc
                eval_prob += prob_acc
            else:
                break
            accuracy = sklearn.metrics.accuracy_score(label.long().cpu().detach().numpy(), pred.cpu().detach().numpy())
            eval_acc += accuracy
        print('Average test acc: {:.6f}'.format(eval_acc/ len(validation_generator)))
        logging.info('Average test acc: {:.6f}'.format(eval_acc/ len(validation_generator)))
        print('Normal test Acc: {:.6f}'.format(eval_norm/ (len(validation_generator))))
        logging.info('Normal test Acc: {:.6f}'.format(eval_norm/ (len(validation_generator))))
        if output_dim == 2:
           print('Malicious test Acc: {:.6f}'.format(eval_mal/ (len(validation_generator))))
           logging.info('Malicious test Acc: {:.6f}'.format(eval_mal/ (len(validation_generator))))
           vl = len(validation_generator)
           logging.info('Normal Class:')
           for k, v in dict(eval_norm_dict).items():
               logging.info(str(k) + ': ' + str(v/vl))
           logging.info('Malicious Class:')
           for k, v in dict(eval_mal_dict).items():
               logging.info(str(k) + ': ' + str(v/vl))
        elif output_dim == 5:
            print('DOS test Acc: {:.6f}'.format(eval_dos/ (len(validation_generator))))
            print('U2R test Acc: {:.6f}'.format(eval_u2r/ (len(validation_generator))))
            print('R2L test Acc: {:.6f}'.format(eval_r2l/ (len(validation_generator))))
            print('PROBE test Acc: {:.6f}'.format(eval_prob/ (len(validation_generator))))

if __name__ == '__main__':                                                      
    train(train_src_iter_Q,train_tgt_iter_Q,train_tgt_iter) 
    #test()                        
