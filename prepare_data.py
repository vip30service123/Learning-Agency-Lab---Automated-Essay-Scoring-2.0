from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split 


def prepare_data(config: DictConfig) -> None:
    print("#### Preparing Data")

    df = pd.read_csv(config.prepare_data.data_path)[['full_text', 'score']]

    train_df, test_df = train_test_split(df, train_size=config.prepare_data.train_size)

    train_df.to_csv(config.prepare_data.train_path)
    test_df.to_csv(config.prepare_data.test_path)

    print("#### End Process")


if __name__=="__main__":
    config = OmegaConf.load("params.yaml")
    prepare_data(config)
