from abc import abstractmethod
from typing import List

from sentiment_analysis.data_processing import DataProcessor


class TextProcessor(DataProcessor):

    def __init__(self,
                 name=None,
                 remove_numbers=False,
                 pos_to_remove=None,
                 remove_stopwords=False,
                 normalize=None):
        """
        Preprocessing abstract class. Treats textual data prior to running a model
        :param name: name of preprocessor (default=class name)
        :param remove_numbers: if to remove numbers from text
        :param remove_stopwords: if to remove stopwords from text
        :param pos_to_remove: list of PoS tags to remove
        :param normalize:  Options=("Stem","Lemmatize"). If to apply stemming or lemmatization"""

        super().__init__()
        if name:
            self._name = name
        else:
            self.name = self.__class__.__name__

        self._remove_numbers = remove_numbers
        self._pos_to_remove = pos_to_remove
        self._remove_stopwords = remove_stopwords
        self._normalize = normalize

    def apply(self, text):
        """
        Returns a clean version of the text string
        :param text: text to preprocess
        :return: a string after preprocessing
        """
        pass

    def apply_batch(self, texts):
        """
        Returns a clean version of the text string
        :param texts: list of texts to preprocess
        :return: a list of strings after preprocessing
        """
        pass

    @abstractmethod
    def tokenize(self, text) -> List[str]:
        """
        Tokenizes the text into tokens
        :param text: Text string
        :return: List[str] containing tokens
        """
        pass

    def __str__(self):
        return f"[name:{self.name}, remove_numbers:{self._remove_numbers}, pos_to_remove:{self._pos_to_remove}," \
               f"remove stopwords:{self._remove_stopwords}, normalize: {self._normalize}]"

    def __repr__(self):
        return self.__str__()
