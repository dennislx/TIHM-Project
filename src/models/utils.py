import inspect
import os
import cmat
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import random

def validate_class(choices):
    def decorator(cls):
        original_init = cls.__init__
        def new_init(self, *args, **kwargs):
            full_args = dict(zip(cls.__init__.__code__.co_varnames[1:], args), **kwargs)
            for k, v in full_args.items():
                if k in choices and k not in choices[k]:
                    raise ValueError(f"Invalid value for '{k}'. Allowed choices are {choices[k]}, but got '{v}'.")
            original_init(self, *args, **kwargs)
        cls.__init__ = new_init
        return cls
    return decorator

def  filter_args(func, kwargs):
    sig = inspect.signature(func)
    param_names = set(sig.parameters)
    filtered_args, unselected_args = {}, {}
    for k,v in kwargs.items():
        if k in param_names: 
            filtered_args[k] = v
        else:
            unselected_args[k] = v
    return filtered_args, unselected_args


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def pjoin(*x, create_if_not_exist=False):
    path = os.path.join(*x)
    if create_if_not_exist:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
    return path

class ConfusionMatrix(cmat.ConfusionMatrix):
    METRICS = ['accuracy', 'precision', 'recall', 'f1score', 'iou', 'rocauc', 'loss']
    @classmethod
    def create(cls, y_true, y_pred, y_prob=None, loss=None, labels=None, names=None ):
        confusion_matrix = cmat.create(y_true, y_pred, labels, names).cmat
        cmatrix = cls(confusion_matrix)
        cmatrix.roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 'nan'
        cmatrix.loss = loss if loss is not None else 'nan'
        return cmatrix
    @property
    def report(self): return {'rocauc': self.roc_auc, 'loss': self.loss, **super().report}


def merge_dict(dict_a, dict_b, prefix_a='a', prefix_b='b'):
    merged_dict = {}
    for key, value in dict_a.items():
        merged_dict[f"{prefix_a}_{key}"] = value
    for key, value in dict_b.items():
        merged_dict[f"{prefix_b}_{key}"] = value
    return merged_dict