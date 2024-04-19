import joblib

from omegaconf import DictConfig, OmegaConf
import pandas as pd

from src.feature_extraction import get_feature_extraction


def extract_feature(config: DictConfig) -> None:
    print("#### Start Extracting Features")

    train_df = pd.read_csv(config.prepare_data.train_path)
    test_df = pd.read_csv(config.prepare_data.train_path)

    vectorizer = get_feature_extraction(config.feature_extraction.vectorizer)

    vectorizer.fit_transform(train_df['full_text'])

    with open(config.feature_extraction.vectorizer_save_path, 'wb') as f:
        joblib.dump(vectorizer, f)

    print("#### End Process")



if __name__=="__main__":
    config = OmegaConf.load("params.yaml")
    extract_feature(config)
