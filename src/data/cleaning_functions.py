import logging

import numpy as np

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin


# -------- IDENTITY TRANSFORM ---------------------
class Identity(BaseEstimator, TransformerMixin):
    '''Passes data through'''

    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# -------- EXTRACT COLUMNS ------------------------
class ExtractCols(BaseEstimator, TransformerMixin):
    '''Extracts columns from a pandas df and returns
    as a np.array'''

    def __init__(self, cols):
        '''Takes in a list of column names'''
        self.cols = cols
        return

    def fit(self, X, y=None):
        return self

    def transform(self, D):
        '''D should be a pandas df'''
        return D[self.cols].values


# -------- STACK COLUMNS --------------------------
class ColumnStacker(BaseEstimator, TransformerMixin):
    '''Stacks columns from ColumnExtractor'''

    def __init__(self):
        return

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.T.ravel()


# -------- SPLIT COLUMNS --------------------------
class Splitter(BaseEstimator, TransformerMixin):
    '''Splits a long matrix in 2 and hstacks the parts'''

    def __init__(self):
        return

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        nrows = x.shape[0]
        return np.hstack((x[:nrows, :], x[nrows:, :]))


# -------- REMOVE UNSIGHTLY CHARACTERS ------------
class CharFilter(BaseEstimator, TransformerMixin):
    '''Filters valid characters.
    If log_filtered_chars = True then we will keep track of
    how many characters are removed and what they are.  Which
    will slow down execution

    -input_dimension: arbitrary,
    -output_dimension: matches input

    -input_type: str
    -output_type: str (containing only valid_chars characters)
    '''

    def __init__(self, valid_chars, log=False):
        self.valid_chars = valid_chars
        self._clean_str = np.vectorize(self._clean_str, otypes=[str])
        self.log = log
        if self.log:
            self.logger = logging.getLogger(__name__)
            self.filtered_chars = {}
        return

    def _clean_str(self, s):
        if self.log:
            ret_str = ''
            for c in str(s):
                if c in self.valid_chars:
                    ret_str += c
                else:
                    try:
                        self.filtered_chars[c] += 1
                    except KeyError:
                        self.filtered_chars[c] = 1
            return ret_str
        else:
            return ''.join(
                [c if c in self.valid_chars else '' for c in str(s)])

    def write_log(self):
        self.logger.info('Filtered characters: %s', self.filtered_chars)
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._clean_str(X)


# -------- TOKENIZE WORDS -------------------------
class WordTokenize(BaseEstimator, TransformerMixin):
    '''Tokenizes strings into a list of (str) words
    -input_dimension: arbitrary,
    -output_dimension: matches input

    -input_type: str
    -output_type: [str]
    '''

    def __init__(self):
        self.wt = np.vectorize(word_tokenize, otypes=[list])
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.wt(X)


# -------- POSITION TAG WORDS --------------------
class POSTag(BaseEstimator, TransformerMixin):
    '''Position tags a list of words, and returns a list of (word, tag)
    tuples.
    -input_dimension: arbitrary,
    -output_dimension: matches input

    -input_type: str
    -output_type: [(str, str)]
    '''

    def __init__(self):
        self.pos_tag = np.vectorize(pos_tag, otypes=[list])
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.pos_tag(X)


# -------- LEMMATIZE POS TAGGED WORDS ------------
class Lemmatize(BaseEstimator, TransformerMixin):
    '''Lemmatizes all the words in an array.
    -input_dimension: arbitrary,
    -output_dimension: matches input

    -input_type: [(str, str)]
    -output_type: str
    '''

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self._lemmatize_str = np.vectorize(self._lemmatize_str, otypes=[list])
        return

    def _get_wordnet_pos(self, treebank_tag):
        '''Converts the treebank POS tag to a tag compaitible with the
        wordnet lemmatizer
        '''
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _lemmatize_str(self, s):
        return ' '.join(
            self.wnl.lemmatize(ti[0], self._get_wordnet_pos(ti[1]))
            for ti in s)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._lemmatize_str(X)
