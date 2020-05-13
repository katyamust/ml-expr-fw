from abc import ABC, abstractmethod


class DataProcessor(ABC):

    def __init__(self):
        """
        Umbrella abstract class for all pre or post data processors.
        Inherit from the class to implement data cleaning, preprocessing or postprocessing
        Treats data prior/after to running a model
        """

    @abstractmethod
    def apply(self, **kwargs):
        pass

    @abstractmethod
    def apply_batch (self, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass