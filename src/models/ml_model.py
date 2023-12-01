import xgboost
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn import svm, linear_model, ensemble, neural_network, neighbors, metrics
import joblib
import models.utils as utils
__all__ = ['XGBoostModel', 'MLPModel', 'RFModel', 'SVMModel', 'LogisticModel', 'GBMModel']




class Model:
    framework = "ML"

    @classmethod
    def create(cls, np_data, target, **args):
        model = cls()
        args, model.clf = model.build(**args)
        X, y = np_data.reshape(len(np_data), -1), target.reshape(-1)
        model.clf.fit(X, y)
        return args, model

    @classmethod
    def restore(cls, path):
        model = cls()
        args, model.clf = joblib.load(path)
        return model

    def save(self, path, args): 
        joblib.dump((args, self.clf), path)

    @property
    def Algorithm(self): raise NotImplementedError

    def calculate_loss(self, y_true, y_pred):
        return metrics.log_loss(y_true, y_pred)

    def build(self, **args):
        args, _ = utils.filter_args(self.Algorithm, args)
        return args, self.Algorithm(**args)

    def predict(self, np_data, target, return_confidence=False, return_loss=False, **kwargs):
        rtn_dict = {'y_pred': None, 'y_prob': None, 'loss': None}
        X, y = np_data.reshape(len(np_data), -1), target.reshape(-1)
        y_hat = self.clf.predict_proba(X)
        rtn_dict['y_pred'] = self.clf.classes_[y_hat.argmax(axis=1)]
        if return_confidence:
            rtn_dict['y_prob'] = y_hat.max(axis=1)
        if return_loss:
            rtn_dict['loss'] = self.calculate_loss(y, y_hat)
        return rtn_dict


class MLPModel(Model):
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    Algorithm = neural_network.MLPClassifier

class XGBoostModel(Model):
    # Reference: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    Algorithm = xgboost.XGBClassifier
    LE = LabelEncoder()

    def train(self, x, y):
        y = self.LE.fit_transform(y)
        super().train(x, y)

    def predict(self, x):
        pred, conf = super().predict(x)
        return self.LE.inverse_transform(pred), conf

class GBMModel( Model ):
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    Algorithm = ensemble.GradientBoostingClassifier

class RFModel(Model):
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Algorithm = ensemble.RandomForestClassifier


class SVMModel(Model):
    # See https://scikit-learn.org/stable/modules/svm.html
    Algorithm = svm.SVC


class KNNModel(Model):
    Algorithm = neighbors.KNeighborsClassifier


class LogisticModel:
    # See: logistic regression sklearn coefficients
    Algorithm = linear_model.LogisticRegression

