from  xgboost import XGBRegressor
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_model(model_name: str):
    if model_name == "xgboost":
        return XGBRegressor()
    elif model_name == "svc":
        return SVC()
    elif model_name == "lightgbm":
        return LGBMClassifier()
    elif model_name == "catboost":
        return CatBoostClassifier()
    else:
        raise Exception("This model is not available.")