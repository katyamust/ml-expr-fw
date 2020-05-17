import re

import nltk
from nltk.corpus import stopwords

from mlfabric.data_processing.text import TextProcessor


class NltkTextProcessor(TextProcessor):

    def __init__(self,
                 remove_numbers=True,
                 pos_to_remove=None,
                 remove_stopwords=True,
                 normalize=None):
        super().__init__(remove_numbers=remove_numbers,
                         pos_to_remove=pos_to_remove,
                         remove_stopwords=remove_stopwords,
                         normalize=normalize)

        self.__wordnet_lemmatizer = nltk.WordNetLemmatizer()
        self.__porter_stemmer = nltk.PorterStemmer()

        # nltk.download('stopwords')
        # nltk.download('wordnet')
        # nltk.download('averaged_perceptron_tagger')

    def preprocess_as_list(self, texts):
        clean_texts = []
        for text in texts:
            clean_texts.append(self.preprocess(text))

        return clean_texts

    def preprocess(self, text):

        # Normalize text
        text = text.lower().strip()

        text = self.tokenize(text)

        # POS Tagging
        if self._pos_to_remove:
            pos = nltk.pos_tag(text)
            text = [x[0] for x in pos if x[1][:2] not in self._pos_to_remove]

        # Remove Numbers
        if self._remove_numbers:
            text = [x for x in text if not re.match(r'''(?x)(?:^[p€$%]*\d+\.*\d*[€$%]*$)''', x)]

        # Remove Stopwords
        if self._remove_stopwords:
            text = [x for x in text if x not in stopwords.words('english')]

        # Normalize
        if self._normalize == "Lemmatize":
            text = [self.__wordnet_lemmatizer.lemmatize(x) for x in text]
        elif self._normalize == "Stem":
            text = [self.__porter_stemmer.stem(x) for x in text]
        else:
            pass

        # Clean
        text = [x.strip() for x in text if x != '']

        return " ".join(text)

    def tokenize(self, text):
        # Tokenize
        pattern = r''' (?x)
                            (?:\w+'\w+)                            # match words separated by a '
                            |(?:\w+(?:-\w+)+)                      # part-one; face-to-face
                            |(?:\w\.)+                             # U.S.A. etch
                            |(?:\w+\.\w+(?:.\w+)+)                 # 12.402.2mvdwu.10.0696cal
                            |(?x)(?:[p%€$.]*\d+[,.]*\d*[%€$]*)     # $21.3 3.56% p0.001 456 4.5 etc.
                            |\w+                                   # match any word
                            |\s+                                   # spaces 
                            '''
        text = nltk.regexp_tokenize(text, pattern)

        # Fixing things the regexp_tokenizer misses:
        word_tokens = []
        for word in text:
            word_tokens.extend(nltk.word_tokenize(word))

        return word_tokens
