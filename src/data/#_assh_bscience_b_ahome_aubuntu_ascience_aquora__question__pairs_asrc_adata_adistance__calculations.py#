import os
import sys
import click
import logging
import nltk
import numpy as np
import pandas as pd

import boto3

from dotenv import get_variable; env_file = '/home/ubuntu/science/quora_question_pairs/.env'
from ipyparallel import Client
from ast import literal_eval
from nltk.metrics import masi_distance, edit_distance, jaccard_distance

S3_BUCKET = get_variable(env_file, 'S3_BUCKET')
S3_DATA_PATH = get_variable(env_file, 'S3_DATA_PATH')
PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = int(get_variable(env_file, 'CHUNKSIZE'))
MASI_DISTANCE = get_variable(env_file, 'MASI_DISTANCE')
EDIT_DISTANCE = get_variable(env_file, 'EDIT_DISTANCE')
JACCARD_DISTANCE = get_variable(env_file, 'JACCARD_DISTANCE')
Q_WORD_TOKENIZED = literal_eval(get_variable(env_file, 'Q_WORD_TOKENIZED'))

def masi_distance_chunk(D):
    '''
    Calculates masi distance between tokenized list pairs
    of questions and saves the result to a column specified by the
    environment variable MASI_DISTANCE
    '''
    if len(D) > 0:
        D[MASI_DISTANCE] = D.loc[:, Q_WORD_TOKENIZED].apply(
            lambda x: masi_distance(set(literal_eval(x[0])),
                                    set(literal_eval(x[1]))),
            axis = 1)
    return D

def edit_distance_chunk(D):
    '''
    Calculates edit distance between tokenized/lemmatized list pairs
    of questions and saves the result to a column specified by the
    environment variable EDIT_DISTANCE
    '''
    if len(D) > 0:
        D[EDIT_DISTANCE] = D.loc[:, Q_WORD_TOKENIZED].apply(
            lambda x: edit_distance(literal_eval(x[0]),
                                    literal_eval(x[1])) / 
            len(literal_eval(x[0])) + len(literal_eval(x[1])),
            axis = 1)
    return D

def jaccard_distance_chunk(D):
    '''
    Calculates the jaccard distance between lemmatized list pairs.
    '''
    if len(D) > 0:
        D[JACCARD_DISTANCE] = D.loc[:, Q_WORD_TOKENIZED].apply(
            lambda x: jaccard_distance(set(literal_eval(x[0])),
                                       set(literal_eval(x[1]))),
            axis = 1)
    return D

@click.command()
@click.argument('test', type = click.Path(), default = 'False')
@click.argument('i_max', type = click.Path(), default = 0)
def main(test, i_max):
    i_max = int(i_max)
    if test == 'True': #Don't chunk
        for f_name in ['train', 'test']:
            print('Calculating Distances (test)', f_name)
            D = pd.read_csv(PROJECT_DIR + '/data/interim/' + f_name + '_test.csv',
                            index_col = 'id')
            D = edit_distance_chunk(D)
            D = masi_distance_chunk(D)
            D = jaccard_distance_chunk(D)
            D.to_csv(PROJECT_DIR + '/data/interim/' + f_name + '_test.csv',
                     index_label = 'id')
    else: #test != 'True'
        pool = Client()
        with pool[:].sync_imports():
            from nltk.metrics import masi_distance, edit_distance
            from nltk.metrics import jaccard_distance
            from ast import literal_eval
        push_res = pool[:].push({'Q_WORD_TOKENIZED' : Q_WORD_TOKENIZED,
                                 'EDIT_DISTANCE' : EDIT_DISTANCE,
                                 'MASI_DISTANCE' : MASI_DISTANCE,
                                 'JACCARD_DISTANCE' : JACCARD_DISTANCE,
                                 'edit_distance_chunk' : edit_distance_chunk,
                                 'masi_distance_chunk' : masi_distance_chunk,
                            'jaccard_distance_chunk' : jaccard_distance_chunk})
        N_JOBS = len(pool)
        left_indices = range(0, CHUNKSIZE, CHUNKSIZE // N_JOBS)
        right_indices = range(CHUNKSIZE // N_JOBS, CHUNKSIZE + 1,
                              CHUNKSIZE // N_JOBS)

        for f_name in ['train', 'test']:
            D_it = pd.read_csv(PROJECT_DIR + '/data/interim/' + f_name + '.csv',
                               chunksize = CHUNKSIZE, index_col = 'id')
            D0 = D_it.get_chunk()
            D0 = edit_distance_chunk(D0)
            D0 = masi_distance_chunk(D0)
            D0 = jaccard_distance_chunk(D0)
            D0.to_csv(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv',
                      mode = 'w', index_label = 'id')
            del D0

            i = 0
            for Di in D_it:
                i += 1
                if i_max != 0 and i > i_max:
                    break
                print('Calculating Distances ',
                      f_name, ' chunk: ', i, end = '\r')
                sys.stdout.flush()
                results = []
                for pi, li, ri in zip(pool, left_indices, right_indices):
                    results.append(pi.apply_async(
                        lambda D: 
                        jaccard_distance_chunk(
                            masi_distance_chunk(
                                edit_distance_chunk(D)
                            )),
                        Di[li:ri]))

                for res in results:
                    Di = res.get()
                    if len(Di) > 0:
                        Di.to_csv(PROJECT_DIR +
                                  '/data/interim/' + 'D_tmp.csv',
                                  mode = 'a', header = False,
                                  index_label = 'id')
                    del Di
            print()
            os.rename(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv',
                      PROJECT_DIR + '/data/interim/' + f_name + '.csv')
    return

if __name__ == '__main__':
  main()
