import os
import sys
import click
import logging
import nltk
import numpy as np
import pandas as pd
import joblib
import time
import boto3

from dotenv import get_variable; env_file = '/home/ubuntu/science/quora_question_pairs/.env'
from ipyparallel import Client
from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

S3_BUCKET = get_variable(env_file, 'S3_BUCKET')
S3_DATA_PATH = get_variable(env_file, 'S3_DATA_PATH')
PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = int(get_variable(env_file, 'CHUNKSIZE'))
MASI_DISTANCE = get_variable(env_file, 'MASI_DISTANCE')
EDIT_DISTANCE = get_variable(env_file, 'EDIT_DISTANCE')
JACCARD_DISTANCE = get_variable(env_file, 'JACCARD_DISTANCE')
Q_WORD_TOKENIZED = literal_eval(get_variable(env_file, 'Q_WORD_TOKENIZED'))
Q_TYPE1=['question1_type1', 'question2_type1']

test_rows = 1000

@click.command()
@click.argument('test', type = click.Path(), default = 'False')
def main(test):
    if test == 'True':
        nrows = test_rows
        sys.stdout.flush()
    else:
        nrows = None

    D_q1 = pd.read_csv(PROJECT_DIR + '/data/processed/test.csv',
                       index_col = 'id', usecols = ['id', Q_WORD_TOKENIZED[0]],
                       nrows = nrows)
    nrows = len(D_q1)
    q1 = D_q1.loc[:, Q_WORD_TOKENIZED[0]].apply(lambda l:
                                                ' '.join(literal_eval(l)))
    del D_q1

    D_q2 = pd.read_csv(PROJECT_DIR + '/data/processed/test.csv',
                       index_col = 'id', usecols = ['id', Q_WORD_TOKENIZED[1]],
                       nrows = nrows)
    q2 = D_q2.loc[:, Q_WORD_TOKENIZED[1]].apply(lambda l: 
                                                ' '.join(literal_eval(l)))
    del D_q2

    all_questions = q1.append(q2)
    all_questions.index = range(len(all_questions))

    t = joblib.load(PROJECT_DIR + '/models/tfidf.pkl')
    print('VECTORIZING...')
    t0 = time.clock()
    tfidf = t.transform(all_questions.values)
    print('Time: ', time.clock() - t0)
    
    nmf_tfidf = joblib.load(PROJECT_DIR + '/models/nmf_tfidf.pkl')
    print('FACTORIZING...')
    t0 = time.clock()
    W = nmf_tfidf.transform(tfidf)
    print('Time: ', time.clock() - t0)
    
    W = np.abs(W[:nrows, :] - W[nrows:, :])

    D = pd.read_csv(PROJECT_DIR + '/data/processed/test.csv', index_col = 'id',
                    usecols = ['id', MASI_DISTANCE,
                               JACCARD_DISTANCE, EDIT_DISTANCE],
                    dtype = np.float64, nrows = nrows)
    Dist = D.values
    del D

    D = pd.read_csv(PROJECT_DIR + '/data/processed/test.csv', index_col = 'id',
                       usecols = ['id', *Q_TYPE1],
                       dtype = 'object', nrows = nrows)
    D = D.loc[:, Q_TYPE1].applymap(literal_eval)
    T = np.hstack((np.vstack(D.loc[:, Q_TYPE1[0]]),
                   np.vstack(D.loc[:, Q_TYPE1[1]])))
    del D

    X = np.hstack((W, Dist, T))
    xgb = joblib.load(PROJECT_DIR + '/models/cv_xgb.pkl')
    y_hat = xgb.predict_proba(X)

    cls1 = np.where(xgb.classes_ == 1)[0][0]
    y_hat = y_hat[:, cls1]
    
    y_hat = pd.DataFrame(y_hat, columns = ['is_duplicate'])
    y_hat.index.name = 'test_id'
    y_hat.to_csv(PROJECT_DIR + '/reports/submissions/cv_xgboost.csv')
    return

if __name__ == '__main__':
  main()
