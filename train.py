import joblib
import time

from omegaconf import DictConfig, OmegaConf
import pandas as pd

from src.model import get_model
from src.utils import running_time


@running_time
def train(config: DictConfig) -> None:
    print("#### Start Training")

    train_df = pd.read_csv(config.prepare_data.train_path)
    # test_df = pd.read_csv(config.prepare_data.train_path)

    with open(config.feature_extraction.vectorizer_save_path, 'rb') as f:
        vectorizer = joblib.load(f)    

    X_train = vectorizer.transform(train_df['full_text'])
    y_train = train_df['score']

    model = get_model(config.training.model)

    model.fit(X_train, y_train)

    with open(config.training.save_model_path, 'wb') as f:
        joblib.dump(model, f)

    print("#### End Training")



if __name__=="__main__":
    config = OmegaConf.load("params.yaml")
    train(config)
