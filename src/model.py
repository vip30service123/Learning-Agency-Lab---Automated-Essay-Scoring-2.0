from  xgboost import XGBRegressor


def get_model(model_name: str):
    if model_name == "xgboost":
        return XGBRegressor()
    else:
        raise Exception("This model is not available.")