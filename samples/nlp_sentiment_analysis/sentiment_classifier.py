import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from mlfabric.models.base_model import BaseModel
from mlfabric.data_processing import EmptyProcessor


class SentimentClassifier(BaseModel):
    """
    Example model implementation to demonstrate the approach
    Model that predicts the movie review text sentiment: is it positive or negative review.
    """

    def __init__(self, model_name='SentimentClassifier',
                 preprocessor=EmptyProcessor(),
                 postprocessor=EmptyProcessor()
                 ):
        """
        :param model_name: name of model
        :param preprocessor: TextPreprocessor object for text data in this example
        :param postprocessor: TextPostprocessor object for text data in this example
        """

        self.vectorizer = TfidfVectorizer()
        self.clf = SGDClassifier()

        super().__init__(model_name=model_name,
                         preprocessor=preprocessor,
                         postprocessor=postprocessor)

    def fit(self, X, y=None) -> None:
        """
        Fits model
        Calculates TF-IDF vectors and fits SGDClassifier model
        :param X: input train data list of movie reviews
        :param y: input train label sentiment for each review
        """
        corpus = self.preprocessor.apply(X)

        logging.info("Finished preprocessing input data, fitting TF IDF vectorizer")

        X_tf_idf = self.vectorizer.fit_transform(corpus)

        logging.info("Fitting classifier")
        self.clf.fit(X_tf_idf, y)

        logging.info("Finished fitting model")

    def predict(self, X) -> pd.DataFrame:
        """
        Predicts movie reviews  sentiment
        :param X: list of of movie text reviews
        :return:  list of sentiment value 0 or 1 for each review
        """
        corpus = self.preprocessor.apply(X)
        Xp_tf_idf = self.vectorizer.transform(corpus)
        y_predicted = self.clf.predict(Xp_tf_idf)
        self.postprocessor.apply(y_predicted)
        return y_predicted
