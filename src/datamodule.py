"""
In this file, the two most important objects are the classes:

- TIHM: The data loading class for the tihm data.

    Activity
        Bathroom | Bedroom | Fridge Door | Hallway | Kitchen | Lounge | Door
        浴室| 卧室| 冰箱门| 走廊| 厨房| 休息室 | 门
        the hourly count of activity in each room per day
    Demographics
    Labels
        Agitation | Blood pressure | Body temperature | Body water | Pulse | Weight
        躁动| 血压| 体温| 身体水分| 脉搏| 体重
    Physiology
        Body Temperature | Body weight | Diastolic blood pressure | Heart rate | O/E - muscle mass | Skin Temperature | Systolic blood pressure | Total body water
        体温| 体重| 舒张压| 心率 | O/E - 肌肉质量 | 皮肤温度 | 收缩压| 体内总水分

- TIHMDataset: The pytorch dataset wrapping the TIHM class.

"""


from __future__ import annotations
import os
import copy
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import operator
import warnings
import functools
import joblib
import utils
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ['AgitationDataset', 'SklearnDataset']

pjoin = lambda *x: os.path.join(*x)
pfold = lambda x: os.path.dirname(x)
pnorm = lambda x: os.path.normpath(x)

class AUGMENT(Enum):
    RAW = 1
    ALL = 2
    RELEVANT = 3

def num_invalid_row(df):
    return sum(df.isna().sum(axis=1) > 0)

def log_summary(data, title, pad='\t'):
    logger.info(f"{pad}{title}")
    logger.info(f"{pad}{pad}sample shape: {data.np_data.shape}")
    logger.info(f"{pad}{pad}class distribution: {np.bincount(data.target.flatten())}")
    
def log_patient(train_patient, test_patient, pad):
    trains, tests = set(train_patient), set(test_patient)
    diff1, diff2, all, common = trains-tests, tests-trains, trains|tests, trains&tests
    logger.info("{}There are {} patients in total with {} appear in both train and test set".format(pad, len(all), len(common)))

def rolling_window(array, window_size):
    assert isinstance(array, (np.ndarray, list)), "Please make sure array is an array or a list"
    assert len(array) >= window_size, "Please make sure the input size >= window size"
    array = np.array(array)
    output = array[ np.lib.stride_tricks.sliding_window_view( np.arange(array.shape[0]), window_shape=window_size) ]
    return output

class StandardGroupScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scalers, self.means_, self.vars_ = {}, {}, {}
        self.global_scalar, self.global_mean_, self.global_var_ = None, None, None
        self.scalars_fitted = False
        self.groups_fitted = []

    def _validate_argument(self, groups, default_shape):
        if groups is None:
            logger.warning('Using sklearn.preprocessing.StandardScaler if groups=None')
            groups = np.ones(default_shape)
        return groups

    def fit(self, X, y, groups=None) -> StandardGroupScaler:
        groups = self._validate_argument(groups, (X.shape[0]))
        self.global_mean_ = np.nanmean(X, axis=0)
        self.global_var_ = np.nanvar(X, axis=0)
        for group_name in np.unique(groups):
            mask = groups == group_name
            X_sub = X[mask]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"Mean of empty slice.*")
                group_means = np.nanmean(X_sub, axis=0)
                warnings.filterwarnings( "ignore", r"Degrees of freedom <= 0 for slice.*")
                group_vars = np.nanvar(X_sub, axis=0)
            replace_with_global_mask = (
                np.isnan(group_means) | np.isnan(group_vars) | (group_vars == 0)
            )
            group_means[replace_with_global_mask] = self.global_mean_[ replace_with_global_mask ]
            group_vars[replace_with_global_mask] = self.global_var_[ replace_with_global_mask ]
            self.means_[group_name] = group_means
            self.vars_[group_name] = group_vars
            self.groups_fitted.append(group_name)
        self.scalars_fitted = True
        return self

    def transform(self, X, y, groups):
        X_norm = copy.deepcopy(X)
        groups = self._validate_argument(groups, (X.shape[0]))
        for group_name in np.unique(groups):
            mask = groups == group_name
            try:
                X_norm[mask] = (X_norm[mask] - self.means_[group_name]) / np.sqrt( self.vars_[group_name])
            except KeyError:
                X_norm[mask] = (X_norm[mask] - self.global_mean_) / np.sqrt( self.global_var_)
        return X_norm

def do_transform(normalize_type, train_X, test_X, train_y, test_y, sample_rate=None) -> Tuple[utils.DataScheme, utils.DataScheme]:
    # step 0: split dataframe into patient, date, and feature
    def do_partition(data, target):
        subject, date = data.pop('patient_id').values, data.pop('date').values
        return subject, date, data.columns, data.values, target.drop(['patient_id', 'date'],axis=1).values
    train_patient, train_date, train_feature, X_train, y_train = do_partition(train_X.copy(), train_y.copy())
    test_patient, test_date, test_feature, X_test, y_test = do_partition(test_X.copy(), test_y.copy())
    # ------------------------------
    # step 1: process missing values
    imputer = SimpleImputer()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    # ------------------------------------------------------------
    # step 2: normalize dataset by either patient id or columnwise
    if normalize_type == "global":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif normalize_type == "id":
        scaler = StandardGroupScaler()
        X_train = scaler.fit_transform(X_train, groups=train_patient)
        X_test = scaler.transform(X_test, groups=test_patient)
    # ----------------------------------------------------
    # step 3: aggregate results and return back dictionary
    train_ds = utils.DataScheme(pd.merge(train_X, train_y, on=['patient_id', 'date']), y_train, train_date, X_train, train_patient, sample_rate, train_feature)
    test_ds = utils.DataScheme(pd.merge(test_X, test_y, on=['patient_id', 'date']), y_test, test_date, X_test, test_patient, sample_rate, test_feature)
    return test_ds, train_ds

def do_rolling(window_length, look_ahead, processed_data):
    patient, date, np_data, target, sample_rate = operator.itemgetter('patient_id', 'date', 'np_data', 'target', 'sample_rate')(processed_data)
    patient_out, np_data_out, date_out, target_out = [], [], [], []
    for p in np.unique(patient):
        idx = np.arange(len(patient))[patient == p][np.argsort(date[patient==p])]
        # ---------------------------------------------------------------------
        # step 1: drop instances where distance between two window is too large
        idx_split = np.where(date[idx][1:]-date[idx][:-1] > pd.to_timedelta(sample_rate))[0]
        idx_split = np.split(np.arange(len(idx)), idx_split + 1)
        for s in idx_split:
            if len(s) < window_length + look_ahead:
                continue # too short to make even one instance
            # ------------------------------------------------------
            # step 2: roll up data with "look ahead" in case predict
            data_s = np_data[idx][s[:len(s)-look_ahead]]
            patient_s = patient[idx][s[:len(s)-look_ahead]]
            date_s = date[idx][s[:len(s)-look_ahead]]
            target_s = target[idx][s[look_ahead:]]
            # ------------------------------------
            # step 3: apply rolling window to data
            data_r, patient_r, date_r, target_r = map( lambda _: rolling_window(_, window_length), (data_s, patient_s, date_s, target_s))
            # ----------------------------
            # step 3: append to the output
            target_r = np.any(~np.isnan(target_r), axis=1).astype(int)
            date_out.append(date_r)
            patient_out.append(patient_r)
            target_out.append(target_r)
            np_data_out.append(data_r)
    processed_data.update(patient_id=np.vstack(patient_out), date=np.vstack(date_out), np_data=np.vstack(np_data_out), target=np.vstack(target_out))
    return processed_data


# def do_augment(augment, data):
#     np_data, feature, date = data['data'], data['feature'], data['date']
#     n_instance, n_window, n_feature = np_data.shape
#     np_data = np_data.reshape(n_instance, -1)
#     first_two_col = pd.DataFrame(index=np.repeat(np.arange(n_instance), n_window)+1, data=dict(time=date.reshape(-1)))
#     rest_col = pd.DataFrame(index=np.repeat(np.arange(n_instance), n_window)+1, data=np_data, columns=feature)
#     prepared_data = pd.concat([first_two_col, rest_col], axis=1).reset_index()
#     f = lambda x, **y: x
#     if augment == AUGMENT.RELEVANT:
#         from tsfresh import extract_relevant_features
#         y = pd.Series(data=np.repeat(data['label'].flatten(), n_window), index=np.repeat(np.arange(n_instance), n_window)+1)
#         f = functools.partial(extract_relevant_features, y=y)
#         logger.info("\tStart extracting relevant augmented features")
#     elif augment == AUGMENT.ALL:
#         from tsfresh import extract_features
#         f = extract_features
#         logger.info("\tStart extracting time augmented features")
#     augment_data = f(prepared_data, column_id='index', column_sort='time')
#     data['data'] = augment_data.drop(['index', 'time'], axis=1).values.reshape(n_instance, n_window, -1)
#     return data


class Instances:

    def __init__(self, data_df, target, date, sample_rate=None, feature_name=None, patient_id=None, np_data=None, sample_weight=None, result_path='./', result_name='baseline', label_info={0: 'NEG', 1: 'POS'}):
        self._data_df = data_df
        self.target = target
        self.date = date
        self.np_data = np_data
        self.sample_rate = sample_rate
        self.sample_weight = sample_weight or compute_sample_weight(class_weight='balanced', y=target)
        self.feature_name = feature_name
        self.patient_id = patient_id
        self.result_path = result_path
        self.result_name = result_name
        self.label_values = list(label_info.keys())
        self.label_names  = list(label_info.values())

    def train_split(self, n_fold, test_window, test_ratio, by='patient'):
        if by == 'patient':
            yield from self.get_patient_split(n_fold, test_ratio)
        elif by == 'time':
            yield from self.get_ts_split(n_fold, test_window)
        
    def get_ts_split(self, n_fold, test_window):
        """ do cross validation evaluation according to TimeSeriesSplit """
        assert isinstance(self.np_data, np.ndarray), f"Please make sure {self.__class__} has prepared dataset"
        end_date = self.end
        for i in range(n_fold):
            test_start = end_date - pd.to_timedelta(test_window)
            train_subsample, test_subsample = self.get_subsample(test_start=test_start)
            yield (i+1, train_subsample, test_subsample)
            end_date = test_start
    
    def get_patient_split(self, n_fold, test_ratio=None):
        """ do cross validation evaluation by patient id """
        from sklearn.model_selection import StratifiedKFold, train_test_split
        p2y = dict(zip(self.patient_id[:, 0], self.target.flatten()))
        patient, target= map(lambda _: np.array(list(_)), (p2y.keys(), p2y.values()))
        if n_fold and n_fold == 1: 
            yield 1, copy.deepcopy(self), copy.deepcopy(self)
        elif n_fold and n_fold > 1: 
            ks = StratifiedKFold(n_fold)
            for i, (_, test_idx) in enumerate(ks.split(patient, target)):
                train_subsample, test_subsample = self.get_subsample(patient[test_idx])
                yield (i+1, train_subsample, test_subsample)
        elif test_ratio is not None and 0 < test_ratio < 1:
            _, test_patient, _, _ = train_test_split(patient, target, test_size=test_ratio)
            train_subsample, test_subsample = self.get_subsample(test_patient)
            yield (1, train_subsample, test_subsample)
        else:
            raise NotImplementedError

    def get_subsample(self, test_patient: np.ndarray = None, test_start: np.datetime64 = None) -> Instances:
        train_ds, test_ds = {}, {}
        if test_start is not None:
            test_mask = self.date[:, 0] >= test_start
            train_ds = utils.DataScheme(self.data_df.query('date < @test_start'), self.target[~test_mask], self.date[~test_mask], self.np_data[~test_mask], patient_id=self.patient_id[~test_mask], sample_rate=self.sample_rate, feature_name=self.feature_name)
            test_ds = utils.DataScheme(self.data_df.query('date <= @test_end & date >= @test_start'), self.target[test_mask], self.date[test_mask], self.np_data[test_mask], patient_id=self.patient_id[test_mask], sample_rate=self.sample_rate, feature_name=self.feature_name)
        if test_patient is not None:
            test_mask = np.isin(self.patient_id[:, 0], test_patient)
            train_ds = utils.DataScheme(self.data_df.query('patient_id not in @test_patient'), self.target[~test_mask], self.date[~test_mask], self.np_data[~test_mask], patient_id=self.patient_id[~test_mask], sample_rate=self.sample_rate, feature_name=self.feature_name)
            test_ds = utils.DataScheme(self.data_df.query('patient_id in @test_patient'), self.target[test_mask], self.date[test_mask], self.np_data[test_mask], patient_id=self.patient_id[test_mask], sample_rate=self.sample_rate, feature_name=self.feature_name)
        return Instances(**train_ds), Instances(**test_ds)
        
    @property
    def data_df(self): return self._data_df
    @property
    def start(self): return self.date.min()
    @property
    def end(self): return self.date.max()
    def __getitem__(self, key): return getattr(self, key)
    def keys(self): return ['np_data', 'date', 'target', 'patient_id', 'feature_name', 'sample_weight']
    def __len__(self): return len(self.target)

    def to_dataloader(self, stage, batch_size, include_time=False, weight_sample=True):
        global BATCH_INFO 
        BATCH_INFO = {'include_time': include_time, 'feature_name': list(self.feature_name), 'weight_sample': weight_sample}
        kwargs = dict(shuffle=True, drop_last=True)
        stage != 'train' and kwargs.update(shuffle=False, drop_last=False)
        return DataLoader(AgitationDataset(self), batch_size=batch_size, collate_fn=AgitationDataset.make_batch, **kwargs)
    
    def to_mldataset(self):
        return SklearnDataset(self.np_data, self.target.squeeze(), self.sample_weight)

class SklearnDataset:

    def __init__(self, np_data, target, sample_weight):
        self.np_data = np_data
        self.target= target
        self.sample_weight = sample_weight

    def keys(self): return ['np_data', 'target', 'sample_weight']

    def __getitem__(self, key): return getattr(self, key)

    def __len__(self): return len(self.target)

    def __add__(self, other: SklearnDataset):
        np_data = np.concatenate((self.np_data, other.np_data), axis=0)
        target = np.concatenate((self.target, other.target), axis=0)
        sample_weight = compute_sample_weight(class_weight='balanced', y=target)
        return SklearnDataset(np_data, target, sample_weight)
    

class AgitationDataset(Dataset):

    def __init__(self, data: Instances):
        self.np_data = data.np_data
        self.target = data.target.squeeze()
        self.dayhour = utils.process_date(pd.to_datetime(data.date), data.sample_rate)
        self.patient = data.patient_id[:, 0]
        self.sampel_weight = data.sample_weight

    def __getitem__(self, index):
        return { 
            'x': self.np_data[index], 'y': self.target[index], 
            'sw': self.sampel_weight[index], 't': self.dayhour[index], 
        }
    
    def __len__(self):
        return len(self.target)

    @staticmethod
    def make_batch(batch):
        feature = np.stack([d['x'] for d in batch])
        target  = np.array([d['y'] for d in batch])
        sample_weight = np.array([d['sw'] for d in batch])
        if BATCH_INFO.get('include_time') == True:
            time = np.stack([d['t'] for d in batch])
            feature = np.concatenate((feature, time), axis=-1)
            BATCH_INFO['feature_name'].extend([f't_time_{i}' for i in range(time.shape[-1])])
        rtn_batch = {
            'x': torch.tensor(feature, dtype=torch.float32),
            'y': torch.tensor(target, dtype=torch.long),
        }
        if BATCH_INFO.get('weight_sample') == True:
            rtn_batch['sw'] = torch.tensor(sample_weight, dtype=torch.float32)
        return rtn_batch

    
class TIHM:

    DATA_NAMES = ["activity", "sleep", "physiology", "labels", "demographics"]
    STATISTICS = ["max", "mean", "std", "sum"]
    TARGET_NAMES = ['Blood pressure', 'Agitation', 'Body water', 'Pulse', 'Weight', 'Body temperature']
    DEFAULT_CACHE_PATH = "{data_cache_dir}/cached/{prefix}-{self.result_name}-f{fold}-l{look_ahead}-n{normalize_type[0]}-s{roll_window}-s{self.sample_rate}-t{test_window}.job"

    def __init__( self, root = "./", result_path = './', target_var='Agitation', include_data = DATA_NAMES, sample_rate=None, **kwargs):
        self.root = root
        self.result_path = result_path
        self._check_exists()
        self._setup()
        d = {k: 0 for k in self.DATA_NAMES if k != 'labels'}
        for data_name in include_data:
            d[data_name] = 1
            setattr(self, f'_include_{data_name}', True)
        self.result_name = ''.join([f'{k[0]}{v}' for k,v in d.items()])
        self.sample_rate = sample_rate
        self._target_types = target_var
        self.date = None
        
    def evaluate_split(self, n_fold, test_window, roll_window, look_ahead, normalize_type='global', prefix='ml', **kwargs):
        self.sample_rate = self.sample_rate or kwargs.get('sample_rate', '1d')
        use_cache = kwargs.get('use_cache', False)
        data_cache_dir = self.result_path
        data_cache_path = kwargs.get('data_cache_path', self.DEFAULT_CACHE_PATH)
        if isinstance(use_cache, str) and os.path.isdir(pjoin(use_cache, 'cached')):
            data_cache_dir = use_cache
        data, target = self.data, self.target
        global logger
        logger = utils.get_logger(name=self.result_path)
        end_date = self.date.max()
        for i in range(n_fold):
            test_start = end_date - pd.to_timedelta(test_window)
            fold = i + 1
            CACHE_PATH = pnorm(data_cache_path.format(**locals()))
            logger.info('-'*100 + f"\nFold {fold}: Starts @ {utils.now()}")
            if use_cache and os.path.exists(CACHE_PATH):
                logger.info("\tLoading preprocessed data from %s"%CACHE_PATH)
                train_ds, test_ds = joblib.load(CACHE_PATH)
            else:
                test_end = test_start + pd.to_timedelta(test_window)
                train_X, test_X = data.query('date < @test_start'), data.query('date <= @test_end & date >= @test_start')
                train_y, test_y = target.query('date < @test_start'), target.query('date >= @test_start & date <= @test_end')
                test_ds, train_ds = do_transform(normalize_type, train_X, test_X, train_y, test_y, self.sample_rate)
                test_ds, train_ds = map(functools.partial(do_rolling, roll_window, look_ahead), (test_ds, train_ds))
                logger.info("Save preprocessed data to %s"%CACHE_PATH)
                os.makedirs(pfold(CACHE_PATH), exist_ok=True)
                joblib.dump((train_ds, test_ds), CACHE_PATH)
            train_data = Instances(**train_ds)
            logger.info(f"Fold {i+1}: Data information: ")
            log_summary(train_data, 'Training Sample:', '\t')
            test_data  = Instances(**test_ds)
            log_summary(test_data, 'Testing Sample:', '\t')
            log_patient(np.unique(train_data.patient_id), np.unique(test_data.patient_id), '\t')
            yield i+1, train_data, test_data
            end_date = test_start

    def _setup(self):
        self._activity_raw = None
        self._activity_df = None
        self._sleep_raw = None
        self._sleep_df = None
        self._physiology_raw = None
        self._physiology_df = None
        self._data_raw = None
        self._data_df = None
        self._target_raw = None
        self._target_df = None
        self._target_types = None
        self._demographic_raw = None
        self._demographic_df = None
        self._demographic_types = None
        self._include_physiology = None
        self._include_activity = None
        self._include_sleep = None
        self._include_demographics = None
    
    def _process_activity(self):
        activity_data = (
            self.activity_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby(["patient_id", pd.Grouper(key="date", freq=self.sample_rate), "location_name"])
            .size()  # counting number of labels of each type
            .unstack()  # long to wide data frame
            .reset_index()
        )
        return activity_data

    def _process_sleep(self):
        sleep = (
            self.sleep_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby( [ "patient_id", pd.Grouper(key="date", freq=self.sample_rate), ])
            .agg({"heart_rate": self.STATISTICS, "respiratory_rate": self.STATISTICS})
        )
        sleep.columns = sleep.columns.map("_".join).str.strip("_")
        sleep = sleep.reset_index()
        return sleep

    def _process_physiology(self):
        physiology = (
            self.physiology_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby(["patient_id", pd.Grouper(key="date", freq=self.sample_rate), "device_type"])
            .agg({"value": self.STATISTICS})
            .unstack()  # long to wide data frame
        )
        physiology.columns = physiology.columns.map("_".join).str.strip("_")
        physiology = physiology.reset_index()
        return physiology

    def _process_target(self) -> pd.DataFrame:
        labels = (
            self.target_raw.assign( date=lambda x: pd.to_datetime(x["date"]))
            .groupby(["patient_id", pd.Grouper(key="date", freq=self.sample_rate), "type"])
            .size()
            .unstack()
            .reset_index()
        )
        return labels
    
    def _process_demographics(self) -> pd.DataFrame:
        return self.demographic_raw

    def __len__(self) -> int:
        return len(self.data)
        
    def _check_exists(self) -> bool:
        return np.all( [ os.path.exists(os.path.join(self.root, f"{name.title()}.csv")) for name in self.DATA_NAMES ])
    
    @property
    def activity_raw(self) -> pd.DataFrame:
        if self._activity_raw is None:
            data_name = "activity"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            self._activity_raw = pd.read_csv(data_path)
        return self._activity_raw
    @property
    def activity(self) -> pd.DataFrame:
        if self._activity_df is None:
            self._activity_df = self._process_activity().sort_values( ["patient_id", "date"])
        return self._activity_df
    @property
    def sleep_raw(self) -> pd.DataFrame:
        if self._sleep_raw is None:
            data_name = "sleep"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            self._sleep_raw = pd.read_csv(data_path)
        return self._sleep_raw
    @property
    def sleep(self) -> pd.DataFrame:
        if self._sleep_df is None:
            self._sleep_df = self._process_sleep().sort_values(["patient_id", "date"])
        return self._sleep_df
    @property
    def physiology_raw(self) -> pd.DataFrame:
        if self._physiology_raw is None:
            data_name = "physiology"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            self._physiology_raw = pd.read_csv(data_path)
        return self._physiology_raw
    @property
    def physiology(self) -> pd.DataFrame:
        if self._physiology_df is None:
            self._physiology_df = self._process_physiology().sort_values( ["patient_id", "date"])
        return self._physiology_df
    @staticmethod
    def merge_all(dataframes, merge_keys=["patient_id", "date"]):
        def cool(df, prefix):
            df.columns = [prefix+col if col not in merge_keys else col for col in df.columns]
            return df
        if not dataframes: return None
        dataframes = [cool(df, p) for df, p in dataframes]
        merged_df = functools.reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how='outer'), dataframes)
        return merged_df.sort_values(merge_keys)
    @property
    def data(self) -> pd.DataFrame:
        if self._data_df is None:
            dataframes = []
            self._include_activity and dataframes.append((self.activity, 'a_'))
            self._include_sleep and dataframes.append((self.sleep, 's_'))
            self._include_demographics and dataframes.append((self.demographic, 'd_'))
            self._include_physiology and dataframes.append((self.physiology, 'p_'))
            self._data_df = self.merge_all(dataframes)
        self.date = self._data_df.date.values
        return self._data_df
    @property
    def data_raw(self) -> pd.DataFrame:
        if self._data_raw is None:
            dataframes = []
            self._include_activity and dataframes.append((self.activity_raw, 'a_'))
            self._include_sleep and dataframes.append((self.sleep_raw, 's_'))
            self._include_demographics and dataframes.append((self.demographic_raw, 'd_'))
            self._include_physiology and dataframes.append((self.physiology_raw, 'p_'))
            self._data_raw = self.merge_all(dataframes)
        return self._data_raw
    @property
    def target_raw(self) -> pd.DataFrame:
        if self._target_raw is None:
            data_name = "labels"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            self._target_raw = pd.read_csv(data_path)
        return self._target_raw
    @property
    def target(self) -> pd.DataFrame:
        if self._target_df is None:
            self._target_df = self._process_target()
            data_labelled = pd.merge( left=self.data, right=self._target_df, how="left", on=["patient_id", "date"],)
            self._target_df = data_labelled[ ["patient_id", "date"] + list(self.target_types) ].sort_values(["patient_id", "date"])
        return self._target_df
    @property
    def target_types(self) -> List[str]:
        if self._target_types is None:
            self._target_types = self.target_raw["type"].unique()  # getting label types
        return self._target_types
    @property
    def demographic_raw(self) -> pd.DataFrame:
        if self._demographic_raw is None:
            data_name = "demographics"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            self._demographic_raw = pd.read_csv(data_path)
        return self._demographic_raw
    @property
    def demographic(self) -> pd.DataFrame:
        if self._demographic_df is None:
            self._demographic_df = self._process_demographics().sort_values( ["patient_id"])
        return self._demographic_df

class DIHM(TIHM):
    
    DATA_NAMES = ["activity", "sleep", "physiology", "labels", "demographics"]
    TARGET_NAMES = ['Blood pressure', 'Agitation', 'Body water', 'Pulse', 'Weight', 'Body temperature']
    
    def evaluate_split(self, n_fold, test_window=7, roll_window=7, look_ahead=0, normalize_type='global',**kwargs):
        yield from super().evaluate_split(n_fold, test_window=test_window, roll_window=roll_window, look_ahead=look_ahead, normalize_type=normalize_type, prefix='dl', **kwargs)

    def overlapping(self):
        """
        Findings:
            1) there exists 135 records of agitation, majority happens at 12:00 and 18:00 (90%...), I think
                they are told to measure at this time
            2) when people in deep and rem mode sleep, they cannot agitate...make sense. there exists two places when people in light mode and agitate. But only 17 people provides sleep data
            4) the lab data is also measured one time each day, to make it less valued in prediction
        """
        def get_time(df): 
            return set(pd.to_datetime(df['date']).dt.floor('H').dt.strftime('%m-%d-%H') + '_' + df['patient_id'])
        target = self.target_raw.query('type == "Agitation"')
        snore  = self.sleep_raw.query('snoring==True')
        # pd.get_dummies(self.sleep_raw['state'])
        sleep  = self.sleep_raw.query('state in ["DEEP", "REM"]')
        light  = self.sleep_raw.query('state == "LIGHT"')
        awake  = self.sleep_raw.query('state == "AWAKE"')
        sta, ssn, ssl, sli, saw = map(get_time, (target, snore, sleep, light, awake))

    @staticmethod
    def display_subdf(df, groupby, query=None):
        df['day'] = pd.to_datetime(df['date']).dt.strftime('%m-%d')
        if query: df = df.query(query)
        for *x, sub in df.groupby(groupby): print(sub)
    
    def _process_activity(self):
        activity_data = (
            self.activity_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby(["patient_id", pd.Grouper(key="date", freq=self.sample_rate), "location_name"])
            .size()  # counting number of labels of each type
            .unstack()  # long to wide data frame
            .reset_index()
        )
        return activity_data

    def _process_sleep(self):
        sleep = (
            self.sleep_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby( [ "patient_id", pd.Grouper(key="date", freq=self.sample_rate), ])
            .agg({"heart_rate": self.STATISTICS, "respiratory_rate": self.STATISTICS})
        )
        sleep.columns = sleep.columns.map("_".join).str.strip("_")
        sleep = sleep.reset_index()
        return sleep

    def _process_physiology(self):
        physiology = (
            self.physiology_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby(["patient_id", pd.Grouper(key="date", freq=self.sample_rate), "device_type"])
            .agg({"value": self.STATISTICS})
            .unstack()  # long to wide data frame
        )
        physiology.columns = physiology.columns.map("_".join).str.strip("_")
        physiology = physiology.reset_index()
        return physiology

    def _process_target(self) -> pd.DataFrame:
        labels = (
            self.target_raw.assign( date=lambda x: pd.to_datetime(x["date"]))
            .groupby(["patient_id", pd.Grouper(key="date", freq=self.sample_rate), "type"])
            .size()
            .unstack()
            .reset_index()
        )
        return labels
    
    def _process_demographics(self) -> pd.DataFrame:
        return self.demographic_raw