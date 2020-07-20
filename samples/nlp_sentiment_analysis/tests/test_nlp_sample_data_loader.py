import pytest

import pandas as pd

from sentiment_analysis.data.nlp_sample_data_loader import NLPSampleDataLoader

def test_can_load():

    my_loader = NLPSampleDataLoader("imdb", 1.0)
    df1, df2 = my_loader.get_dataset()

    x = df1['text'][198]
    
    assert x is not None