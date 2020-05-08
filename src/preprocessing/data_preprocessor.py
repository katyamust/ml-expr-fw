from abc import ABC, abstractmethod


class DataPreprocessor(ABC):

    def __init__(self):
        """
        Umbrella abstract class for all pre or post data processors.
        Treats data prior/after to running a model
        """

    @abstractmethod
    def preprocess(self, **kwargs):
        pass

    @abstractmethod
    def preprocess_as_list (self, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass