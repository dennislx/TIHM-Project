from .ml_model import *
from .dl_model import *

def get_model(name):
    if name == 'MLP':
        return MLPModel
    elif name == 'LSTM':
        return LSTMModel
    elif name == 'GBM':
        return GBMModel
    else: 
        raise ValueError( f"Model {name} not implemented !!!" )