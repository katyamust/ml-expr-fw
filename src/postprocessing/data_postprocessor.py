from abc import ABC, abstractmethod


class DataPostprocessor(ABC):

    def __init__(self):
        """
        Umbrella abstract class for all post data processors
        """

    @abstractmethod
    def postprocess(self, **kwargs):
        pass

    @abstractmethod
    def postprocess_as_list (self, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass