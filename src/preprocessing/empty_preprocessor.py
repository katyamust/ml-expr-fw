from continuous_review.text_preprocessing.preprocessor import Preprocessor


class EmptyPreprocessor(Preprocessor):
    """
    Preprocessor that doesn't do a thing to the text.
    """

    def preprocess_text(self, text):
        return text

    def preprocess_text_list(self, texts):
        return texts

    def tokenize(self, text):
        return text.split(" ")
