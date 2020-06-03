import pandas as pd

from .data_loader import DataLoader

class NLPSampleDataLoader(DataLoader):
    def download_dataset(self) -> None:
        pass

    def get_dataset(self):
        df_train = pd.read_csv('../csv/imdb_train.csv')
        df_test  = pd.read_csv('../csv/imdb_test.csv')

        return df_train, df_test
