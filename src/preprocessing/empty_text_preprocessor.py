from src.preprocessing.text_preprocessor import TextPreprocessor


class EmptyTextPreprocessor(TextPreprocessor):
    """
    Preprocessor that doesn't do a thing to the text.
    """

    def preprocess(self, text):
        return text

    def preprocess_as_list(self, texts):
        return texts

    def tokenize(self, text):
        return text.split(" ")
