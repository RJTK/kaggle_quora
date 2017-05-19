import sys
import click
import numpy as np
import pandas as pd
import joblib
import time

from ast import literal_eval

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

n_components = 20  #Number of NMF components
max_tfidf_features = 15000  #Number of features for tfidf
test_rows = 1000


class ColumnExtractor(BaseEstimator, TransformerMixin):
    '''Extracts a column from a pandas .csv'''

    def __init__(self, file_name, col_name, nrows=None):
        self.file_name = file_name
        self.col_name = col_name
        return

    def fit(self, x, y=None):
        return self

    def transform(self, x=None):
        D = pd.read_csv(file_name, usecols=[col_name], nrows=nrows)
        return D.values.ravel()


class ColumnStacker(BaseEstimator, TransformerMixin):
    '''Stacks columns from ColumnExtractor'''

    def __init__(self):
        return

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.T.ravel()


@click.command()
@click.argument('test', type=click.Path(), default='False')
def main(test):
    if test == 'True':
        nrows = test_rows
        sys.stdout.flush()
    else:
        nrows = None

    file_name = PROJECT_DIR + '/data/processed/train.csv'
    extract_q1 = ColumnExtractor(file_name, Q_WORD_TOKENIZED[0], nrows=nrows)
    extract_q2 = ColumnExtractor(file_name, Q_WORD_TOKENIZED[1], nrows=nrows)
    q_stacker = ColumnStacker()
    pipeline = FeatureUnion(
        [('extract_q1', extract_q1), ('extract_q2', extract_q2)], n_jobs=2)

    pipeline = Pipeline([('question_extractor', pipeline), ('q_stacker',
                                                            q_stacker)])

    D_q1 = pd.read_csv(
        PROJECT_DIR + '/data/processed/train.csv',
        index_col='id',
        usecols=['id', Q_WORD_TOKENIZED[0]],
        nrows=nrows)
    nrows = len(D_q1)
    q1 = D_q1.loc[:, Q_WORD_TOKENIZED[0]].apply(
        lambda l: ' '.join(literal_eval(l)))
    del D_q1

    D_q2 = pd.read_csv(
        PROJECT_DIR + '/data/processed/train.csv',
        index_col='id',
        usecols=['id', Q_WORD_TOKENIZED[1]],
        nrows=nrows)
    q2 = D_q2.loc[:, Q_WORD_TOKENIZED[1]].apply(
        lambda l: ' '.join(literal_eval(l)))
    del D_q2

    all_questions = q1.append(q2)
    all_questions.index = range(len(all_questions))

    t = TfidfVectorizer(
        max_df=.95,
        min_df=2,
        stop_words='english',
        max_features=max_tfidf_features,
        ngram_range=(1, 2))

    t0 = time.clock()
    print('VECTORIZING...')
    tfidf = t.fit_transform(all_questions.values)
    print('Time: ', time.clock() - t0)
    joblib.dump(t, PROJECT_DIR + '/models/' + 'tfidf.pkl')

    nmf_tfidf = NMF(n_components=n_components, init='nndsvda')
    print('NMF tfidf...')
    t0 = time.clock()
    W = nmf_tfidf.fit_transform(tfidf)
    print('Time: ', time.clock() - t0)
    joblib.dump(nmf_tfidf, PROJECT_DIR + '/models/' + 'nmf_tfidf.pkl')

    W = np.abs(W[:nrows, :] - W[nrows:, :])

    D = pd.read_csv(
        PROJECT_DIR + '/data/processed/train.csv',
        index_col='id',
        usecols=['id', MASI_DISTANCE, JACCARD_DISTANCE, EDIT_DISTANCE],
        dtype=np.float64,
        nrows=nrows)
    Dist = D.values
    del D

    D = pd.read_csv(
        PROJECT_DIR + '/data/processed/train.csv',
        index_col='id',
        usecols=['id'] + Q_TYPE1,
        dtype='object',
        nrows=nrows)
    D = D.loc[:, Q_TYPE1].applymap(literal_eval)
    T = np.hstack((np.vstack(D.loc[:, Q_TYPE1[0]]),
                   np.vstack(D.loc[:, Q_TYPE1[1]])))
    #    nmf_T = NMF(n_components = 25, init = 'nndsvda')
    #    print('NMF T...')
    #    t0 = time.clock()
    #    T = nmf_T.fit_transform(T)
    #    print('Time: ', time.clock() - t0)
    #    joblib.dump(nmf_T, PROJECT_DIR + '/models/' + 'nmf_T.pkl')
    del D

    X = np.hstack((W, Dist, T))

    D = pd.read_csv(
        PROJECT_DIR + '/data/processed/train.csv',
        index_col='id',
        usecols=['id', 'is_duplicate'],
        dtype=np.float64,
        nrows=nrows)
    y = D.values.ravel()

    param_dist = {
        'max_depth': range(15),
        'learning_rate': np.logspace(-3, 1, 50),
        'subsample': [0.3, 0.5, 0.7, 1.0],
        'n_estimators': [250, 400, 700, 1000, 1500, 2000],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': np.logspace(-2, 2, 50),
    }

    cv = RandomizedSearchCV(
        XGBClassifier(nthread=2), param_dist, scoring='neg_log_loss', n_jobs=4)

    print('Fitting xgb')
    cv.fit(X, y)
    clf = cv.best_estimator_
    joblib.dump(clf, PROJECT_DIR + '/models/' + 'cv_xgb.pkl')
    return


if __name__ == '__main__':
    main()
