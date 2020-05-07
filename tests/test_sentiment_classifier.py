import pytest

import pandas as pd

from src.models.sentiment_classifier import SentimentClassifier

@pytest.fixture
def mock_train_df():
    return pd.DataFrame (
        {
          'text':  ['excellent movie', 'did not like it', 'horrible actors'],
          'label': ['1', '0', '0']
        },
        columns = ['text','label'])

def mock_test_df():
    return pd.DataFrame( { 'text': ['what I can say abou thsi moview', 'ha ha ha']}, columns=['text'])


def test_sentiment_classifier_can_fit(mock_train_df):

    model = SentimentClassifier()
    model.fit(mock_train_df)

    assert model.X_tf_idf is not None and model.clf is not None


def test_sentiment_classifier_predict(mock_train_df, mock_test_df):
    model = SentimentClassifier()
    model.fit(mock_train_df)
    y = model.predict(mock_test_df)

    assert y[0] == 0






