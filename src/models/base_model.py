import logging
import pickle
from abc import ABC, abstractmethod

from src.data_processing import DataProcessor


class BaseModel(ABC):
    """
    Abstract class for a model with unified interface
    """

    def __init__(self,
                 model_name=None,
                 preprocessor: DataProcessor = None,
                 postprocessor: DataProcessor = None,
                 **hyper_params):
        """
        :param model_name: Model name, to be used by the experiment manager
        :param preprocessor: Preprocessor object that would preprocess each input sample
        :param postprocessor: Postprocessor object that would postprocess data after training/inference
        :param hyper_params: any specific parameter for the model should passed here to be logged
        """
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.__class__.__name__

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.hyper_params = hyper_params

        logging.info(
             f"Created model {self.model_name} "
             f"and hyperparams {self.hyper_params}")

    @abstractmethod
    def fit(self, X, y=None) -> None:
        """
        Trains/fits a model
        :param X: train data
        :param y: optional label
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts with fitted model
        @param X: inference data
        """
        pass

    def get_hyperparams(self):
        """
        Return the model hyper parameters for logging purposes
        :return:
        """
        try:
            return self.hyper_params
        except AttributeError:
            logging.error("Model does not have defined hyper parameters")

    def save(self, file_path: str):
        """
        Stores a model in a pickle. Note that some objects are not pickable.
        In such case the save method should be overridden.
        :param file_path: Path to pickle
        :return:
        """
        with open(file_path, 'wb+') as f:
            pickle.dump(self, file=f)

    @classmethod
    def load(cls, file_path):
        """
        Loads a model from pickle. Note that some objects are not pickable.
        In such case the load method should be overridden.
        :param file_path: Path to pickle file
        :return: An model of type BaseModel
        """
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj
