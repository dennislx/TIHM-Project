import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
import models.utils as utils
from tqdm import tqdm
from enum import Enum
import functools

__all__ = ['LSTMModel']

# Must implement: 
#   create:  return `model_args` that creates its unique ID, and model instance
#   fit:     return training and validation result (ConfusionMatrix) with feeded train_dl, val_dl
#   save:    save training argument as well as training checkpoint
#   restore: load saved model and prepare for evaluation
#   predict: return testing result (dict) with feeded test_dl
class BaseModel:
    framework = "DL"

    @classmethod
    def create(cls, **args):
        # return model_args that makes this configuration different from others in grid_search
        myself = cls()
        myself_args, myself.trainer = myself.build( **args )
        return myself_args, myself

    @classmethod
    def restore(cls, path):
        model = cls()
        train_args, model_params = operator.itemgetter('train_args', 'state_dict')(torch.load(path))
        _, model.trainer = model.build(**train_args)
        model.trainer.load_model(model_param=model_params)
        return model

    def save(self, path, args): 
        self.trainer.save_model(path, args)

    def fit(self, train_dl, val_dl): 
        return self.trainer.train(train_dl, val_dl)

    def predict(self, test_dl): 
        return self.trainer.predict(test_dl)

    @property 
    def Algorithm(self): raise NotImplementedError
    @property
    def Trainer(self): raise NotImplementedError


    def build(self, **args):
        model_args, args = utils.filter_args(self.Algorithm, args)
        model = self.Algorithm(**model_args)
        train_args, args = utils.filter_args(self.Trainer, args)
        trainer = self.Trainer(model=model, **train_args)
        return {**model_args, **train_args}, trainer

class Action(Enum):
    Save = 1
    Stop = 2
    Ignore = 3

class EarlyStopping:
    
    def __init__(self, mode, patience=None, delta=0, save_path=None):
        self.mode = mode
        self.save_path = save_path
        self.delta = delta
        self.max_patience = patience or float('inf')
        self.reset()
    
    def reset(self, score=None):
        self.best_score = score or float('inf') if self.mode == 'min' else float('-inf')
        self.patience_cnt = 0

    def should_stop(self, new_score):
        if self.mode == 'max' and new_score > self.best_score + self.delta:
            self.reset(new_score)
            return Action.Save
        elif self.mode == 'min' and new_score < self.best_score - self.delta:
            self.reset(new_score)
            return Action.Save
        elif self.patience_cnt >= self.max_patience:
            return Action.Stop
        else:
            self.patience_cnt += 1
            return Action.Ignore

def move_to_device(dict_data, device):
    for k, v in dict_data.items():
        if isinstance(v, torch.Tensor):
            dict_data[k] = v.to(device)
    
class EpochData:

    def __init__(self):
        self.data = dict( y_true=[], y_prob=[], y_pred=[])
        self.loss = {'total': 0.0, 'cnt': 0}

    @property
    def report(self):
        loss = self.loss['total']/self.loss['cnt']
        return utils.ConfusionMatrix.create(**self.data, loss=loss)
        
    def add(self, y_true=None, y_prob=None, y_pred=None, loss=None):
        isinstance(y_true, list) and self.data['y_true'].extend(y_true)
        isinstance(y_prob, list) and self.data['y_prob'].extend(y_prob)
        isinstance(y_pred, list) and self.data['y_pred'].extend(y_pred)
        if isinstance(loss, float):
            self.loss['total'] += loss
            self.loss['cnt'] += 1

class Trainer:

    def __init__(self, model, lr, epoch, num_class, device, metric, mode, patience=0, threshold=0.5):
        self.lr = lr
        self.epoch = epoch
        self.num_class = num_class
        self.device = device
        self.metric = metric
        self.model = model
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        global LOSS_FN 
        LOSS_FN = nn.CrossEntropyLoss(reduction='none')
        
    @staticmethod
    def calculate_loss(y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = torch.ones_like(y_true).to(y_true)
        loss = LOSS_FN(y_pred, y_true)
        return (loss * sample_weight / sample_weight.sum()).sum()
    
    def one_batch(self, batch_d):
        saved_d = {'y_true': batch_d.get('y'), 'sample_weight': batch_d.get('sw')}
        batch_d, _ = utils.filter_args(self.model.forward, batch_d)
        self.device != 'cpu' and move_to_device(batch_d, self.device)
        self.device != 'cpu' and move_to_device(saved_d, self.device)
        model_output = self.model(**batch_d).squeeze()
        # ------------- Probably I need to implement the following a lot more flexible ------------
        if model_output.ndim == 1: # in case there is only one sample
            model_output = model_output.unsqueeze(0)
        loss = self.calculate_loss(y_pred=model_output, **saved_d)
        y_prob = F.softmax(model_output, dim=1)[:, 1]         # y_prob = torch.sigmoid(model_output)
        # -----------------------------------------------------------------------------------------
        saved_d = dict(loss=loss.item(), y_true=saved_d['y_true'].tolist(), y_prob=y_prob.tolist(), y_pred=model_output.argmax(axis=1).tolist())
        return loss, saved_d

    def train(self, train_dl, val_dl):
        earlystop = EarlyStopping(self.mode, self.patience)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        device = torch.device('cuda:' + str(self.device) if torch.cuda.is_available() else 'cpu')
        tempsaver = utils.TempFile()
        self.model.to(device)
        train_results, valid_results = [], []
        pbar = tqdm(range(self.epoch))
        for i in pbar:
            pbar.set_description(f'Epoch {i}')
            self.model.train()
            train_res = EpochData()
            for batch in train_dl:
                loss, pred_ds = self.one_batch(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_res.add(**pred_ds)
            train_results.append(train_res.report)
            valid_results.append(self.predict(val_dl))
            train_val_res = utils.merge_dict(train_results[i].report, valid_results[i].report, 'train', 'valid')
            pbar.set_postfix(train_loss=train_val_res['train_loss'], val_loss=train_val_res['valid_loss'])
            next_step = earlystop.should_stop(train_val_res[self.metric])
            if next_step == Action.Save:
                self.save_model(tempsaver.path, {'epoch': i})
            elif next_step == Action.Stop:
                break
        best_epoch = self.load_model(tempsaver.path)['epoch']
        tempsaver.close()
        return train_results[best_epoch], valid_results[best_epoch]

    def predict(self, test_dl):
        device = torch.device('cuda:' + str(self.device) if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        epoch_data = EpochData()
        for batch in test_dl:
            with torch.no_grad():
                _, pred_ds = self.one_batch(batch)
                epoch_data.add(**pred_ds)
        return epoch_data.report
    
    def load_model(self, model_path=None, model_param=None):
        args = {}
        if model_param is None:
            args, model_param = operator.itemgetter('train_args', 'state_dict')(torch.load(model_path))
        self.model.load_state_dict(model_param)
        return args

    def save_model(self, model_path, args={}):
        torch.save({ 'state_dict': self.model.state_dict(), 'train_args': args, 'device': self.device}, model_path)


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_size, out_dim=2, bidirectional=True, num_layers=1, dropout=0.):
        super().__init__()
        self.lstm = nn.LSTM( input_dim, hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        self.last = nn.Sequential( nn.Dropout(dropout), nn.Linear(hidden_size * (1+bidirectional), out_dim) )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = x[-1, :, :]     
        return self.last(x)


class LSTMModel( BaseModel ):
    Algorithm = LSTM
    Trainer = Trainer
    
class AutoEncode(nn.Module):
    
    def __init__(self, input_dim, sequence_length, represent_dim, out_dim=2, dropout=0.):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU()),
            nn.Sequential(nn.Linear(16*sequence_length, 4*sequence_length), nn.ReLU()),
            nn.Sequential(nn.Linear(4*sequence_length, represent_dim), nn.ReLU())
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Linear(represent_dim, 4*sequence_length), nn.ReLU()),
            nn.Sequential(nn.Linear(4*sequence_length, 16*sequence_length), nn.ReLU()),
            nn.Sequential(nn.Linear(16, input_dim), nn.ReLU())
        ])
        self.sequence_length = sequence_length
        self.fc1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(represent_dim, out_dim))
        self.fc2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim*sequence_length, out_dim))
        self.layernorm = nn.LayerNorm([sequence_length, input_dim])
        self.representation = None

    
    def forward(self, x):
        N = len(x)
        x = self.layernorm(x)
        x_e = self.encoder[0](x)
        x_e = self.encoder[1](x_e.reshape(N, -1))
        self.representation = z = self.encoder[2](x_e)
        x_d = self.decoder[0](z)
        x_d = self.decoder[1](x_d)
        x_d = self.decoder[2](x_d.reshape(N, self.sequence_length, -1))
        x_d = self.layernorm(x_d)
        return self.fc2(x_d.reshape(N, -1)) - self.fc2(x.reshape(N, -1)) + self.fc1(z)
    
class NOTWORKING(nn.Module):
    
    def __init__(self, input_dim, sequence_length, represent_dim, out_dim=2, dropout=0.):
        super().__init__()
        WINDOW = max(4, sequence_length // 2); PAD = WINDOW - 2  # out_dim = input_dim + (PAD-1) 
        def f(I): return (I+PAD-1)//2+1
        RELU = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.unpool = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(input_dim, 32, kernel_size=WINDOW, bias=False, padding=PAD), nn.BatchNorm1d(32), RELU), 
            nn.Sequential(nn.Conv1d(32, represent_dim, kernel_size=WINDOW, bias=False, padding=PAD), nn.BatchNorm1d(represent_dim), RELU) 
        ])
        self.represent_dim = f(f(sequence_length)) * represent_dim
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(represent_dim, 32, kernel_size=WINDOW, bias=False, padding=PAD), nn.BatchNorm1d(represent_dim), nn.ReLU()),
            nn.Sequential(nn.Conv1d(32, input_dim, kernel_size=WINDOW, bias=False, padding=PAD), nn.BatchNorm1d(input_dim), nn.ReLU())
        ])
        self.sequence_length = sequence_length
        self.fc1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.represent_dim, out_dim))
        self.fc2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, out_dim))
        self.representation = None
    

class AEModel( BaseModel ):
    Algorithm = AutoEncode
    Trainer = Trainer

    
class CNN(nn.Module):

    def __init__(self, input_dim, sequence_length, hidden_dim, out_dim=2, dropout=0.):
        super().__init__()
        WINDOW = max(4, sequence_length // 2); PAD = WINDOW - 2  # out_dim = input_dim + (PAD-1) 
        f = lambda I: (I+PAD-1)//2+1
        if not isinstance(hidden_dim, (tuple, list)):
            hidden_dim = [hidden_dim]
        self.num_block = len(hidden_dim)
        self.represent_dim = functools.reduce(lambda x, _: f(x), range(self.num_block), sequence_length) * hidden_dim[-1]
        blocks = list()
        in_channel = input_dim
        for out_channel in hidden_dim:
            blocks.append(nn.Conv1d(in_channel, out_channel, kernel_size=WINDOW, bias=False, padding=PAD))
            blocks.append(nn.BatchNorm1d(out_channel))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
            in_channel = out_channel
        self.cnn = nn.Sequential(*blocks)
        self.last = nn.Sequential( nn.Dropout(dropout), nn.Linear(self.represent_dim, out_dim) )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.reshape(-1, self.represent_dim)
        return self.last(x)


class CNNModel( BaseModel ):
    Algorithm = CNN
    Trainer = Trainer