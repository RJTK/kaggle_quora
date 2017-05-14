import os
import sys
import click
import logging
import nltk
import boto3

import numpy as np
import pandas as pd

from dotenv import get_variable; env_file = '/home/ubuntu/science/quora_question_pairs/.env'
from ipyparallel import Client
from ast import literal_eval

S3_BUCKET = get_variable(env_file, 'S3_BUCKET')
S3_DATA_PATH = get_variable(env_file, 'S3_DATA_PATH')
PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = int(get_variable(env_file, 'CHUNKSIZE'))
Q = literal_eval(get_variable(env_file, 'Q'))
Q_WORD_TOKENIZED = literal_eval(get_variable(env_file, 'Q_WORD_TOKENIZED'))

def tokenize_chunk(Di):
    '''Word tokenize chunks'''
    if len(Di) > 0:
        Di.loc[:, Q] = Di.loc[:, Q].applymap(str) #Ensure we have strings
        Di[Q_WORD_TOKENIZED] = Di.loc[:, Q].applymap(nltk.word_tokenize)
    return Di

@click.command()
@click.argument('test', type = click.Path(), default = 'False')
@click.argument('i_max', type = click.Path(), default = 0)
def main(test, i_max):
    i_max = int(i_max)
    if test == 'True': #Don't chunk
        for f_name in ['train', 'test']:
            print('Tokenizing (test)', f_name)
            D = pd.read_csv(PROJECT_DIR + '/data/interim/' + f_name + '_test.csv',
                            index_col = 'id')
            D = tokenize_chunk(D)
            D.to_csv(PROJECT_DIR + '/data/interim/' + f_name + '_test.csv',
                     index_label = 'id')

    else: #test != 'True'
        pool = Client()
        with pool[:].sync_imports():
            import nltk
        push_res = pool[:].push({'Q': Q,
                                 'Q_WORD_TOKENIZED': Q_WORD_TOKENIZED})

        N_JOBS = len(pool)
        left_indices = range(0, CHUNKSIZE, CHUNKSIZE // N_JOBS)
        right_indices = range(CHUNKSIZE // N_JOBS, CHUNKSIZE + 1,
                              CHUNKSIZE // N_JOBS)

        for f_name in ['train', 'test']:
            D_it = pd.read_csv(PROJECT_DIR + '/data/interim/' + f_name + '.csv',
                               chunksize = CHUNKSIZE, index_col = 'id')
            D0 = D_it.get_chunk()
            D0 = tokenize_chunk(D0)
            D0.to_csv(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv',
                      mode = 'w', index_label = 'id')
            del D0

            i = 0
            for Di in D_it:
                i += 1
                if i_max != 0 and i > i_max:
                    break
                print('Tokenizing ', f_name, ' chunk: ', i, end = '\r')
                sys.stdout.flush()
                results = []
                for pi, li, ri in zip(pool, left_indices, right_indices):
                    results.append(pi.apply_async(tokenize_chunk,
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
