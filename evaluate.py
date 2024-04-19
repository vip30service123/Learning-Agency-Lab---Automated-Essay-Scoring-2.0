import joblib
import time

from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils import running_time


@running_time
def evaluate(config: DictConfig) -> None:
    print("#### Start Evaluating")

    test_df = pd.read_csv(config.prepare_data.train_path)

    with open(config.feature_extraction.vectorizer_save_path, 'rb') as f:
        vectorizer = joblib.load(f)    

    X_test = vectorizer.transform(test_df['full_text'])
    y_test = test_df['score']

    model = joblib.load(config.training.save_model_path)

    y_pred = model.predict(X_test)
    y_pred = [int(i) for i in y_pred]

    evaluate_score = cohen_kappa_score(y_pred, y_test, weights='quadratic')

    with open(config.evaluation.save_result_path, 'w') as f:
        f.write(str(evaluate_score))

    print("#### End Evaluating")


if __name__=="__main__":
    config = OmegaConf.load("params.yaml")
    evaluate(config)
