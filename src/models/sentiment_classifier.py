import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from src.models.base_model import BaseModel
from src.preprocessing.empty_text_preprocessor import EmptyTextPreprocessor
from src.postprocessing.empty_postprocessor import EmptyPostprocessor


class SentimentClassifier(BaseModel):
    """
    Example model implementation to demonstrate the approach
    Model that predicts the movie review text sentiment: is it positive or negative review.
    """

    def __init__(self, model_name='SentimentClassifier',
                 preprocessor=EmptyTextPreprocessor(),
                 postprocessor= EmptyPostprocessor(),
                 feature_set=None):
        """
        :param model_name: name of model
        :param preprocessor: TextPreprocessor object for text data in this example
        :param postprocessor: TextPostprocessor object for text data in this example
        :param feature_set: specific set of features for this example model implementation
        """

        if feature_set is None:
            feature_set = {'x_column_name': 'text', 'y_label_name': 'label'}
        self.feature_set = feature_set

        self.vectorizer = TfidfVectorizer()
        self.X_tf_idf = None
        self.clf = None

        super().__init__(model_name=model_name,
                         preprocessor=preprocessor,
                         postprocessor=postprocessor,
                         feature_set=self.feature_set)
        print(self.hyper_params)


    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits model using pandas DataFrame as input dataset
        Calculates TF-IDF vectors and fits SGDClassifier model
        :param df: input dataset with 2 columns ["text"] and ["value"]
        """

        imdb_train_df = df
        corpus = self.preprocessor.preprocess(imdb_train_df[self.feature_set["x_column_name"]].values)

        logging.info("Finished preprocessing input data, fitting TF IDF vectorizer")

        self.X_tf_idf = self.vectorizer.fit_transform(corpus)

        logging.info("Fitting classifier")
        self.clf = SGDClassifier()
        Y_train = imdb_train_df[self.feature_set["y_label_name"]].values
        self.clf.fit(self.X_tf_idf, Y_train)

        logging.info("Finished fitting model")

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts movie review movie sentiment
        :param df_test: pandas DataFrame of movie text reviews, consists of 1 column ["text"]
        """
        imdb_test_df = df_test
        corpus = imdb_test_df[self.feature_set["x_column_name"]]

        X_test = self.vectorizer.transform(corpus)

        y_predict = self.clf.predict(X_test)

        self.postprocessor.postprocess(y_predict)
        return y_predict
