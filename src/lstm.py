"""
The record is defined as
    28 sensor data + one date information + 6 label information + one patient

The date information:
    2019-04-01 => 2019-06-14 || 2019-06-15 => 2019-06-30
    75 training dates || 16 testing dates

The target information:
    ['Blood pressure', 'Agitation', 'Body water', 'Pulse', 'Weight', 'Body temperature']
    躁动| 血压| 体温| 身体水分| 脉搏| 体重
    we have 474 records with >= 1 labels
            34  records with >= 2 labels
            1   record  with >= 3 labels

The patient information:
    56 patients with number of records per each patient:
    3.0 -> 39.0    39.0 -> 54.5    54.5 -> 68.0    68.0 -> 91.0
    -------------  --------------  --------------  --------------
            14              14              13              15
            
The input shapes are:
    we have 2032 records for training
            767  records for testing
with n_days=7 rolling window:
    we have 1690 instances for training
            422  instances for testing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torchdata

from sklearn.utils import compute_sample_weight
from sklearn import metrics

# usual imports
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

# loading dataset
from datautils import TIHMDataset
DPATH = 'data/'

class AgitationDataset(torchdata.Dataset):
    def __init__(self, dataset):
        self.data, self.target = [], []
        for x, y in dataset:
            self.data.append(torch.tensor(x).float())            
            self.target.append(np.int64(y[-1, 1]>=1)) # the last of the days and agitation
        # also define the sample weight
        self.sw = compute_sample_weight(class_weight='balanced', y=self.target)

    def __getitem__(self, index):
        return self.data[index], self.target[index], self.sw[index]
    
    def __len__(self):
        return len(self.data)

def load_data(is_train=True, normalise='global', n_days=7, bs = 100, test_start="2019-06-23"):
    """
    @param normalise:   global | id,      standardize data globally or group by patient id
    @n_days:    create a rolling window of original sequence
                e.g., [1,2,3,4], n_days=3 => [1,2,3], [2,3,4]
    """
    dataset = TIHMDataset(root=DPATH, train=is_train, normalise=normalise, n_days=n_days, test_start=test_start)
    dataset = AgitationDataset(dataset)
    return torchdata.DataLoader(dataset=dataset, batch_size=bs, shuffle=is_train)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(num_features=sequence_length),
            nn.ReLU(),
            )
        self.lstm = nn.LSTM( input_dim, hidden_size, batch_first=True,)
        self.last = nn.Sequential( nn.Linear(hidden_size, 2))

    def forward(self, x):
        x = self.fc(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :] # get the value from the last of the sequence
        x = self.last(x)
        return x

def train(model, criterion, optimiser, train_loader, n_epochs):
    training_loss = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    def one_batch(x, y, sw):
        x, y, sw = x.to(device), y.to(device), sw.to(device)
        criterion.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss = (loss * sw / sw.sum()).sum()
        loss.backward()
        optimiser.step()
        return loss
    def one_epoch(train_loader):
        epoch_loss = []
        for batch in train_loader:
            loss = one_batch(*batch)
            epoch_loss.append(loss.item())
        return epoch_loss
    for _ in tqdm.tqdm(range(n_epochs), desc='Training'):
        epoch_loss = one_epoch(train_loader)
        training_loss.extend(epoch_loss)
    return training_loss

def predict(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    ypred, ytrue = [], []
    for x, y, _ in test_loader:
        x = x.to(device)
        outputs = model(x)
        ypred.append(F.softmax(outputs, dim=1).detach().cpu())
        ytrue.append(y)
    return torch.cat(ypred).numpy(), torch.cat(ytrue).numpy()


train_dl, test_dl = load_data(test_start="2019-06-23"), load_data(is_train=False, test_start="2019-06-23")
_, L, E = next(iter(train_dl))[0].shape
lstm = LSTMModel(input_dim=E, sequence_length=L, hidden_size=64)
loss = train(
    model=lstm, 
    criterion=nn.CrossEntropyLoss(reduction='none'), 
    optimiser=torch.optim.Adam(lstm.parameters(), lr=0.001),
    train_loader=train_dl,
    n_epochs=50,
)
pred, target = predict(lstm, test_dl)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(target, pred.argmax(axis=1)))
print(classification_report(target, pred.argmax(axis=1)))
import cmat
print(cmat.create(target, pred.argmax(axis=1)).report)