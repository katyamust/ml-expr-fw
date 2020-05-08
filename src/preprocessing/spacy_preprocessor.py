import re
from typing import List

import spacy
from spacy.language import Language
from tqdm import tqdm

from continuous_review.text_preprocessing import Preprocessor


class SpacyPreprocessor(Preprocessor):

    @staticmethod
    def download_spacy_model(model="en_core_web_sm"):
        spacy.cli.download(model)

    @staticmethod
    def load_model(model="en_core_web_sm"):
        return spacy.load(model, disable=["ner", "parser"])

    def tokenize(self, text):
        doc = self.model(text)
        return [token.text for token in doc]

    def __init__(self,
                 spacy_model: Language = None,
                 remove_numbers=True,
                 pos_to_remove=None,
                 remove_stopwords=True,
                 normalize=None):
        super().__init__(remove_numbers=remove_numbers,
                         pos_to_remove=pos_to_remove,
                         remove_stopwords=remove_stopwords,
                         normalize=normalize)

        if not spacy_model:
            self.model = spacy.load("en_core_web_sm")
        else:
            self.model = spacy_model

    def preprocess_text(self, text):
        doc = self.model(text)
        return self.__clean(doc)

    def preprocess_text_list(self, texts=List[str]):
        clean_texts = []
        for doc in tqdm(self.model.pipe(texts)):
            clean_texts.append(self.__clean(doc))

        return clean_texts

    def __clean(self, doc):

        tokens = []
        # POS Tagging
        if self._pos_to_remove:
            for token in doc:
                if token.pos_ not in self._pos_to_remove:
                    tokens.append(token)
        else:
            tokens = doc

        # Remove Numbers
        if self._remove_numbers:
            tokens = [token for token in tokens if not (token.like_num or token.is_currency)]

        # Remove Stopwords
        if self._remove_stopwords:
            tokens = [token for token in tokens if not token.is_stop]
        # remove unwanted tokens
        tokens = [token for token in tokens if
                  not (token.is_punct or token.is_space or token.is_quote or token.is_bracket)]

        # Remove empty tokens
        tokens = [token for token in tokens if token.text.strip() != ""]

        # Normalize
        if self._normalize == "Lemmatize":
            text = " ".join([token.lemma_ for token in tokens])
        elif self._normalize == "Stem":
            raise ValueError("spaCy does not have a stemmer")
        else:
            text = " ".join([token.text for token in tokens])

        text = re.sub(r'[^a-zA-Z\']', ' ', text)
        # remove Unicode characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        text = text.lower()

        return text
