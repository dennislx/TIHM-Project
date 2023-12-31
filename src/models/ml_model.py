import xgboost
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn import svm, linear_model, ensemble, neural_network, neighbors, metrics
import joblib
import inspect
import models.utils as utils
__all__ = ['XGBoostModel', 'MLPModel', 'RFModel', 'SVMModel', 'LogisticModel', 'GBMModel', 'MLModel']




class MLModel:
    framework = "ML"

    def __init__(self, weight_sample, **kwargs):
        self.weight_sample = weight_sample

    @classmethod
    def create(cls, weight_sample, **args):
        model = cls(weight_sample)
        args, model.clf = model.build(**args)
        return args, model
    
    def fit(self, np_data, target, sample_weight):
        X, y = np_data.reshape(len(np_data), -1), target.reshape(-1)
        if self.weight_sample and ('sample_weight' in inspect.signature(self.clf.fit).parameters):
            self.clf.fit(X, y, sample_weight=sample_weight)
        else:
            self.clf.fit(X, y)

    @classmethod
    def restore(cls, path):
        model_args, clf = joblib.load(path)
        model = cls(**model_args)
        model.clf = clf
        return model_args, model

    def save(self, path, args): 
        args.update(weight_sample=self.weight_sample)
        joblib.dump((args, self.clf), path)

    @property
    def Algorithm(self): raise NotImplementedError

    def calculate_loss(self, y_true, y_pred, sample_weight):
        if not self.weight_sample: sample_weight = None
        return metrics.log_loss(y_true, y_pred, sample_weight=sample_weight, labels=[0, 1])

    def build(self, **args):
        args, _ = utils.filter_args(self.Algorithm, args)
        return args, self.Algorithm(**args)

    def predict(self, np_data, target, sample_weight, return_confidence=False, return_loss=False, **kwargs):
        rtn_dict = {'y_pred': None, 'y_prob': None, 'loss': None}
        X, y = np_data.reshape(len(np_data), -1), target.reshape(-1)
        y_hat = self.clf.predict_proba(X)
        rtn_dict['y_pred'] = self.clf.classes_[y_hat.argmax(axis=1)]
        if return_confidence:
            rtn_dict['y_prob'] = y_hat[:, 1]
        if return_loss:
            rtn_dict['loss'] = self.calculate_loss(y, y_hat, sample_weight)
        return rtn_dict


class MLPModel(MLModel):
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    Algorithm = neural_network.MLPClassifier

class XGBoostModel(MLModel):
    # Reference: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    Algorithm = xgboost.XGBClassifier
    LE = LabelEncoder()

    def train(self, x, y):
        y = self.LE.fit_transform(y)
        super().train(x, y)

    def predict(self, x):
        pred, conf = super().predict(x)
        return self.LE.inverse_transform(pred), conf

class GBMModel( MLModel ):
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    Algorithm = ensemble.GradientBoostingClassifier

class RFModel(MLModel):
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Algorithm = ensemble.RandomForestClassifier


class SVMModel(MLModel):
    # See https://scikit-learn.org/stable/modules/svm.html
    Algorithm = svm.SVC


class KNNModel(MLModel):
    Algorithm = neighbors.KNeighborsClassifier


class LogisticModel ( MLModel ):
    # See: logistic regression sklearn coefficients
    Algorithm = linear_model.LogisticRegression

