import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .base_model import BaseModel


class SentimentClassifier(BaseModel):
    """
    Model that predicts the movie review text sentiment: is it positive or negative review.
    """

    def __init__(self, model_name='SentimentClassifier',
                 feature="text"):
        """
        :param model_name: Name of model for logging and experimentation purposes
        :param feature: which feature to calculate TF/IDF for: "text",
        potentially we may have other features
        """
        self.vectorizer = TfidfVectorizer()
        self.feature = feature
        self.X_tf_idf = None
        self.clf = None
        hyper_params = {"feature": feature}
        super().__init__(model_name=model_name, hyper_params=hyper_params)
        logging.info(f"Created model {self.model_name} ")

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits model using pandas DataFrame as input dataset
        Calculates TF-IDF vectors and fits SGDClassifier model
        :param df: input dataset with 2 columns ["text"] and ["value"]
        """

        imdb_train_df = df
        corpus = imdb_train_df[self.feature].values

        logging.info("Finished preprocessing papers, fitting TF IDF vectorizer")

        self.X_tf_idf = self.vectorizer.fit_transform(corpus)

        logging.info("Fitting classifier")
        self.clf = SGDClassifier()
        Y_train = imdb_train_df['label'].values
        self.clf.fit(self.X_tf_idf, Y_train)

        logging.info("Finished fitting vectorizer")

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts movie review movie sentiment
        :param df_test: pandas DataFrame of movie text reviews, consists of 1 column ["text"]
        """
        imdb_test_df = df_test
        corpus = imdb_test_df[self.feature]

        X_test = self.vectorizer.fit_transform(corpus)

        y_predict = self.clf.predict(X_test)

        return y_predict
