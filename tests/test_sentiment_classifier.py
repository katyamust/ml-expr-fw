import pytest

import pandas as pd

from src.models.sentiment_classifier import SentimentClassifier
from src.preprocessing.spacy_text_preprocessor import SpacyTextPreprocessor


@pytest.fixture
def mock_train_df():
    return pd.DataFrame (
        {
          'text':  ['excellent movie', 'did not like it', 'horrible actors'],
          'label': [1, 0, 0]
        },
        columns=['text','label'])


@pytest.fixture
def mock_test_df():
    return pd.DataFrame({'text': ['What I can say about this movie', 'ha ha ha']}, columns=['text'])


def test_sentiment_classifier_can_fit(mock_train_df):

    model = SentimentClassifier()
    X = mock_train_df["text"].values
    y = mock_train_df["label"].values
    model.fit(X,y)

    assert pytest.approx(model.clf.coef_[0][0], 0.001) == -6.93


def test_sentiment_classifier_predict(mock_train_df,mock_test_df):
    model = SentimentClassifier()
    X = mock_train_df["text"].values
    y = mock_train_df["label"].values
    model.fit(X,y)
    Xp = mock_test_df["text"].values
    y = model.predict(Xp)

    assert y[0] == 1


def test_sentiment_classifier_can_fit_with_spacy_preprocessor(mock_train_df):

    preprocessor = SpacyTextPreprocessor()
    model = SentimentClassifier(preprocessor)
    X = mock_train_df["text"].values
    y = mock_train_df["label"].values
    model.fit(X,y)

    assert pytest.approx(model.clf.coef_[0][0], 0.001) == -6.93




