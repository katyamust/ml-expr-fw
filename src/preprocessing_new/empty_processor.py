from src.preprocessing_new import DataProcessor


class EmptyProcessor(DataProcessor):
    """
    Data processor that doesn't do anything to the data
    """

    def apply(self, X):
        return X

    def apply_batch(self, X):
        return X

    def __str__(self):
        pass

    def __repr__(self):
        pass
