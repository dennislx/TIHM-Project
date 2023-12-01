import sys
import os
import cmat
import joblib
import logging
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from datetime import datetime
from omegaconf import OmegaConf, ListConfig
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass, field 
from marshmallow_dataclass import class_schema
from marshmallow import validate as V
from collections import Counter, defaultdict


now = lambda: datetime.now().strftime('%d/%m at %H:%M:%S')

def pjoin(*x, create_if_not_exist=False):
    path = os.path.join(*x)
    if create_if_not_exist:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
    return path

def f(*x):
    return [tuple(y) for y in x]
OmegaConf.register_new_resolver("list.tuple", lambda *x: list(map(tuple, x)))

def one_hot(x, range):
    return np.take(np.eye(range), x, axis=0)

def process_date(date, sample_rate):
    def process_hour(row):
        a, b, c = rate
        if c == 'T': return one_hot((row.minute+row.hour)//a, b)
        return one_hot(row.hour//a, b)
    def process_day(row):
        return one_hot(row.day_of_week, 7)
    def process_rate(r):
        num, unit = int(r[:-1]), r[-1]
        if unit == 'T': return num, 24*60//num, 'T' # minute
        elif unit == 'h': return num, 24//num, 'h'  # hour
        else: return 0, 0, 0
    rate = process_rate(sample_rate)
    np_rtn  = np.array(list(map(process_day, date)))
    if rate[1] > 0: 
        granuity = np.array(list(map(process_hour, date)))
        np_rtn = np.concatenate((np_rtn, granuity), axis=-1)
    return np_rtn

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
    
class Recorder:

    INDEX =  ['seed', 'fold', 'stage', 'model']
    RESULT = ['accuracy', 'precision', 'recall', 'f1score', 'iou']
    
    def __init__(self, name, logs=[], index=INDEX): 
        self.name = name
        self.logs = logs
        self.index = index
        
    def log(self, info, index):
        self.logs.append((index, info))
        
    @classmethod
    def restore(cls, path):
        name, index, logs = joblib.load(path)
        return cls(name=name, logs=logs, index=index)
    
    def get_result(self, train=False, valid=False, test=False):
        if train: yield from self.fetch(stage='train')
        if valid: yield from self.fetch(stage='valid')
        if test:  yield from self.fetch(stage='test')
        
    def fetch(self, return_key=False, **query):
        rtn_results = []
        for index, info in self.logs:
            if all(index.get(k) == v for k, v in query.items()):
                if return_key:
                    remain_key = {k: v for k, v in index.items() if k not in query}
                    rtn_results.append((remain_key, info))
                else:
                    rtn_results.append(info)
        return rtn_results
    
    def save(self, path, index=None):
        if index is None:
            joblib.dump((self.name, self.INDEX, self.logs), path)
        else:
            logs_to_save = self.fetch(return_key=True, **index)
            joblib.dump((self.name, index, logs_to_save), path)
            
    def summary(self, path=None, **query):
        to_summary = self.fetch(return_key=True, **query)
        df = pd.DataFrame([a|b for a,b in to_summary])
        path and df.to_csv(index=False, path_or_buf=path)
        return df

def aggregate_metrics(df, group_by_columns, exclude_columns=[]):
    exclude_columns = exclude_columns + group_by_columns
    agg_columns = [col for col in df.columns if df[col].dtype.kind in 'biufc' and col not in exclude_columns]
    mean_df = df.groupby(group_by_columns)[agg_columns].mean()
    std_df  = df.groupby(group_by_columns)[agg_columns].std()
    merged_df = pd.concat([mean_df, std_df], keys=['mean', 'std'], axis=1)
    merged_df = merged_df.map(lambda x: f'{x:.3f}')
    merged_df = merged_df['mean'] + '(' + merged_df['std'] + ')'
    return merged_df
        
def summarize(runs, logname, savepath):
    # pd.read_csv(savepath, index=[0])
    res_df = pd.concat([r.summary() for _,_,r in runs])
    print("*"*80 + "\nAll running results: ")
    agg_df = aggregate_metrics(res_df, ['model', 'stage'], ['fold', 'seed'])
    res_df.to_csv(savepath)
    logger = get_logger(logname)
    logger.info("*"*200 + "Final Result\n" + agg_df.to_markdown())
    

class ModelSelect:
    
    def __init__(self, mode=None, metric=None, **kwargs):
        self.mode = mode
        self.metric = metric
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.current_score = None

    def is_better(self, result_dict, metric):
        new_score = result_dict[metric]
        if self.mode == 'max' and new_score > self.best_score:
            self.best_score = new_score
            return True
        elif self.mode == 'min' and new_score < self.best_score:
            self.best_score = new_score
            return True
        self.current_score = new_score
        return False
    
    def prepare(self, metric):
        rtn_dict = dict(return_confidence=True, return_loss=True)
        _, metric = metric.split('_')
        if 'aucroc' == metric: rtn_dict['return_confidence'] = True
        if 'loss' == metric: rtn_dict['return_loss'] = True
        return rtn_dict

def merge_dict(dict_a, dict_b, prefix_a='a', prefix_b='b'):
    merged_dict = {}
    for key, value in dict_a.items():
        merged_dict[f"{prefix_a}_{key}"] = value
    for key, value in dict_b.items():
        merged_dict[f"{prefix_b}_{key}"] = value
    return merged_dict

class Averager:
    def __init__(self):
        self.cumsum, self.count = defaultdict(float), defaultdict(int)

    def add(self, dict_values):
        for key, value in dict_values.items():
            if value == 'nan': continue
            self.cumsum[key] += value
            self.count[key] += 1
    @property
    def average(self): 
        return {k: v/self.count[k] if v != 'nan' else v for k,v in self.cumsum.items()}
    
def copy_dict(dict_a, dict_b, *keys):
    rtn_dict = copy.deepcopy(dict_a)
    for key in keys:
        rtn_dict[key] = dict_b.get(key, dict_a[key])
    return rtn_dict

def cross_validation(data, model_cls, model_args, model_path, config, save_intermediate=None):
    ms = ModelSelect(**config.MODEL_SELECTION)
    train_result, val_result, all_cm_result = {}, {}, []
    model_args = {**config.get(model_cls.framework)['Train'], **model_args}
    for i, train_args in enumerate(grid_search(model_args)):
        train_score, val_score = Averager(), Averager()
        split_args = config.TRAIN_SPLIT.to_dict(n_fold=train_args.get('n_fold'), avoid_none=True)
        select_metric = train_args.get('metric', ms.metric)
        predict_args = ms.prepare(select_metric)
        for j, train, val in data.train_split(**split_args):
            if model_cls.framework == 'ML':
                model_arg, model = model_cls.create(**train, **train_args)
                train_report = evaluate(model, train, **predict_args)
                val_report = evaluate(model, val, **predict_args)
            elif model_cls.framework == 'DL':
                BATCH_INFO = {'batch_size': config.BATCH_SIZE}
                train_dl, val_dl = train.to_dataloader(shuffle=True, **BATCH_INFO), val.to_dataloader(**BATCH_INFO)
                _, sequence_length, input_dim = next(iter(train_dl))['x'].shape
                train_args.update(input_dim=input_dim, sequence_length=sequence_length)
                model_arg, model = model_cls.create(**train_args)
                train_report, val_report = model.fit(train_dl, val_dl)
            train_score.add(train_report.report)
            val_score.add(val_report.report)
            if save_intermediate is not None:
                all_cm_result.append((dict(split=j, stage='train', **model_arg), train_report.cmat))
                all_cm_result.append((dict(split=j, stage='valid', **model_arg), val_report.cmat))
        merged_score = merge_dict(train_score.average, val_score.average, 'train', 'valid')
        if ms.is_better(merged_score, metric=select_metric):
            model.save(path=model_path, args=train_args)
            train_result, val_result = train_score.average, val_score.average
    if save_intermediate:
        joblib.dump(all_cm_result, save_intermediate)
    fit_model = model_cls.restore(model_path)
    return fit_model, train_result, val_result

def evaluate(model, data, return_report=False, **evaluate_args):
    if model.framework == 'ML':
        output = model.predict(**data, **evaluate_args)
        to_rtn = ConfusionMatrix.create(data.target.squeeze(), **output)
    elif model.framework == 'DL':
        to_rtn = model.predict(data.to_dataloader(batch_size=512))
    return to_rtn.report if return_report else to_rtn

def save_command(path):
    with open(path, 'a') as f:
        args = [arg if " " not in arg else f'"{arg}"' for arg in sys.argv]
        f.write(f'{now()}\npython {" ".join(args)}\n\n')

def grid_search(args):
    args = args if isinstance(args, (list, tuple, ListConfig)) else [args]
    return ParameterGrid([
        {k: v if isinstance(v, (list, tuple, ListConfig)) else [v] for k, v in a.items()}
        for a in args
    ])

def get_logger(name, savepath=None, debug_mode=False, filemode='append'):
    if debug_mode: loglevel = logging.DEBUG
    else:          loglevel = logging.INFO
    logger = logging.getLogger(name=name)
    logger.setLevel(loglevel)
    loglevel = logging.getLevelName(logger.getEffectiveLevel())
    logger.setLevel(loglevel)
    if savepath is not None:
        fh = logging.FileHandler(savepath)
        fh.setLevel(loglevel)
        formatter = logging.Formatter( '[%(levelname)s] %(message)s' )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.debug(f'Setting log level to: "{loglevel}"') 
    return logger

def load_results(folder):
    results = {}
    try:
        for f in os.listdir(folder):
            if f.startswith('arg') and f.endswith('.job'):
                arg, result = joblib.load(pjoin(folder, f))
                results[arg] = result
    except Exception:
        return 

class Config:

    def __init__(self, config_path):
        self.config = load_configs(config_path)
        os.makedirs(self.RESULT, exist_ok=True)
        save_command(pjoin(self.RESULT, 'command.txt'))
        load_configs(pjoin(self.RESULT, 'config.yaml'), self.config)

    def __getattribute__(self, name):
        try: 
            return super().__getattribute__(name) 
        except AttributeError:
            return getattr(super().__getattribute__('config'), name)

    def get(self, key, default=None):
        return getattr(self.config, key, default)


class BaseScheme:
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): return setattr(self, key, value)
    def keys(self): return self.__dict__.keys()
    def update(self, **kwargs): 
        for k,v in kwargs.items(): setattr(self, k, v)
    def to_dict(self, avoid_none=False, **kwargs):
        merged_dict = dict(self) | kwargs
        if avoid_none:
            for k,v in kwargs.items():
                if v is None: merged_dict[k] = self[k]
        return merged_dict
        
@dataclass
class ResultScheme(BaseScheme):
    score:      float
    argument:   dict
    cm:         cmat.ConfusionMatrix

@dataclass
class DataScheme(BaseScheme):
    data_df:        pd.DataFrame
    target:         np.ndarray
    date:           np.ndarray
    np_data:        np.ndarray
    patient_id:     np.ndarray
    sample_rate:    Optional[str]
    feature_name:   List[str]

@dataclass
class TestSplitScheme(BaseScheme):
    look_ahead: int 
    roll_window: int
    sample_rate: str
    test_window: str = '1d'
    n_fold: int = 1
    normalize_type: str = 'global'
    use_cache: bool = True

@dataclass
class TrainSplitScheme(BaseScheme):
    by: str = field(metadata={'validate': V.OneOf(['patient', 'time'])})
    n_fold: Optional[int] = None
    test_ratio: Optional[float] = None
    test_window: Optional[str] = None

metric_regex = fr'^(train|valid|test)_({"|".join(ConfusionMatrix.METRICS)})$'
@dataclass
class ModelSelectScheme(BaseScheme):
    mode: str = field(metadata={'validate': V.OneOf(['min', 'max'])})
    metric: str = field(metadata={'validate': V.Regexp(metric_regex)})

@dataclass
class EarlyStopScheme(BaseScheme):
    mode: str = field(metadata={'validate': V.OneOf(['min', 'max'])})
    metric: str = field(metadata={'validate': V.Regexp(metric_regex)})
    patience:  Optional[int]
    min_delta: Optional[float]

@dataclass
class FrameDataScheme(BaseScheme):
    target_var: List[str]
    include_data: List[str]
    sample_rate: Optional[str] = None
    
@dataclass
class FrameworkScheme(BaseScheme):
    Data:   FrameDataScheme
    Train:  Optional[Dict] = field(default_factory=lambda: {})

@dataclass
class ConfigScheme(BaseScheme):
    DPATH: str
    RESULT: str
    RUNS: Optional[List[Dict]]
    SEED: List[int]
    TRAIN_SPLIT: TrainSplitScheme
    TEST_SPLIT: TestSplitScheme
    LOGGING: str = field(metadata={'validate': V.OneOf(['append', 'reset'])})
    MODEL_SELECTION: ModelSelectScheme
    DL: Optional[FrameworkScheme]
    ML: Optional[FrameworkScheme]
    EARLY_STOPPING: Optional[EarlyStopScheme]
    TEST_DAYS: Optional[str] 
    WINDOW_SIZE: Optional[int] 
    SKIP_TRAIN: Optional[List[str]]
    SKIP_TEST: Optional[List[str]]
    BATCH_SIZE: Optional[int] = 64
    SAMPLE_RATE: Optional[str] = '1d'
    INCLUDE_DATA: List[str] = field(default_factory=lambda: ['activity'])
    TARGET_VAR: List[str] = field(default_factory=lambda: ['Agitation'])
    EXP_NAME: Optional[str] = 'baseline'

    
def load_configs(path, obj=None) -> ConfigScheme:
    Schema = class_schema(ConfigScheme)
    if obj is not None: 
        assert isinstance(obj, ConfigScheme)
        OmegaConf.save(obj, path)
    config = OmegaConf.load(path)
    return Schema().load(config, unknown='raise')