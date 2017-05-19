import string
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.metrics import masi_distance, edit_distance, jaccard_distance
from nltk.util import ngrams, skipgrams
from sklearn.base import BaseEstimator, TransformerMixin


def skipgram_analyzer(s, skipgram_list=[(1, 0)]):
    '''
    An analyzer that splits a string s into a list of (n, k) skipgrams
    for each (n, k) pair in the skipgrams list.

    Because skipsgrams also produce all the ngrams without skips, it
    is very easy for this function to produce repeated data.  One must
    be careful of what is passed in as the skipgram list.  A good
    start would be [(1, 0), (2, 3), (3, 2)].  But notice that in [(1,
    0), (2, 0), (2, 3)] the tuple (2, 0) is redundant and shouldn't be
    passed in.
    '''
    s = word_tokenize(s.lower())
    s = list(filter(lambda c: c not in string.punctuation, s))

    ret = []
    for n, k in skipgram_list:
        if k == 0:
            ret += list(ngrams(s, n))
        else:
            ret += skipgrams(s, n, k)
    return ret


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


# -------- TOKENIZE SENTENCES ---------------------
class SentTokenize(BaseEstimator, TransformerMixin):
    '''Tokenizes strings into a list of (str) sentences
    -input_dimension: arbitrary,
    -output_dimension: matches input

    -input_type: str
    -output_type: [str]
    '''

    def __init__(self):
        self.st = np.vectorize(sent_tokenize, otypes=[list])
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.st(X)


# -------- INFER QUESTION TYPES ------------------
class QuestionTypes(BaseEstimator, TransformerMixin):
    '''Attempts to identify the 'type' of a question (who, what...)
    by checking for the presence of keywords (specified by the
    'question_types' dictionary) in sentences ending in a '?'
    -input_dimension: arbitrary,
    -output_dimension: matches input

    -input_type: [str]
    -output_type: np.array(int)
    '''

    def __init__(self, question_types):
        # We include a 'type' in which there are no question_type
        # words present.  Hence the + 1
        self.n_types = len(set(question_types.values())) + 1
        ordered_values = list(set(question_types.values()))
        # Ensure the question types map simply to numbers so that
        # we can form an indexed array
        self.question_types = {
            t: ordered_values.index(question_types[t]) + 1
            for t in question_types.keys()
        }
        # self._get_qtype = np.vectorize(self._get_qtype, otypes=[tuple])
        return

    def _get_qtype(self, l):
        '''l should be a list of sentences.  We extract the ones
        which have a '?' in them and then check for the presence
        of the question_type words'''
        qtype = np.array([0] * self.n_types)
        for q in [s if '?' in s else '' for s in l]:
            for word in q.split():
                word = word.lower()
                word = word.split("'")[0]
                if word in self.question_types:
                    qtype[self.question_types[word]] = 1

        if sum(qtype) == 0:  # typeless questions
            qtype[0] = 1
        return qtype

    def _combine_qtypes(self, row):
        # Combines the two qtype arrays for each question
        return np.hstack((self._get_qtype(row[0]),
                          self._get_qtype(row[1])))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.apply_along_axis(self._combine_qtypes, 1, X)
        return np.vstack(X)  # Merge everything


# -------- CALCULATE MASI DISTANCE ---------------
class MasiDistance(BaseEstimator, TransformerMixin):
    '''Calculates the masi_distance between lemmatized list pairs
    -input_dimension: n x 2
    -output_dimension: n x 1

    -input_type: str x str (lemmatized)
    -output_type: np.float64
    '''

    def __init__(self):
        self._masi_distance = np.vectorize(
            self._masi_distance, otypes=[np.float64])
        return

    def _masi_distance(self, s1, s2):
        return masi_distance(set(s1), set(s2))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._masi_distance(s1=X[:, 0], s2=X[:, 1])[:, np.newaxis]


# -------- CALCULATE EDIT DISTANCE ---------------
class EditDistance(BaseEstimator, TransformerMixin):
    '''
    Calculates the edit distance between lemmatized list pairs

    -input_dimension: n x 2
    -output_dimension: n x 1

    -input_type: str x str (lemmatized)
    -output_type: int
    '''

    def __init__(self, sub_cost=1.0, transpositions=False):
        self._edit_distance = np.vectorize(self._edit_distance, otypes=[int])
        self.sub_cost = sub_cost
        self.transpositions = transpositions
        return

    def _edit_distance(self, s1, s2):
        return edit_distance(
            s1,
            s2,
            substitution_cost=self.sub_cost,
            transpositions=self.transpositions)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._edit_distance(s1=X[:, 0], s2=X[:, 1])[:, np.newaxis]


# -------- CALCULATE JACCARD DISTANCE ------------
class JaccardDistance(BaseEstimator, TransformerMixin):
    '''
    Calculates the Jaccard distances between lemmatized list pairs

    -input_dimension: n x 2
    -output_dimension: n x 1

    -input_type: str x str
    -output_type: np.float64
    '''

    def __init__(self):
        self._jaccard_distance = np.vectorize(
            self._jaccard_distance, otypes=[np.float64])
        return

    def _jaccard_distance(self, s1, s2):
        return jaccard_distance(set(s1), set(s2))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._jaccard_distance(s1=X[:, 0], s2=X[:, 1])[:, np.newaxis]
