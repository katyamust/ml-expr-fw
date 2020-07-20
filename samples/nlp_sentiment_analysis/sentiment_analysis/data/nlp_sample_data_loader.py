import pandas as pd

from .data_loader import DataLoader


class NLPSampleDataLoader(DataLoader):
    def download_dataset(self) -> None:
        pass

    def get_dataset(self):
        df_train = pd.read_csv("../data/raw/imdb_train.data")
        df_test = pd.read_csv("../data/raw/imdb_test.data")

        return df_train, df_test
