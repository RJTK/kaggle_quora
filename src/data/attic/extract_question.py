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

S3_BUCKET = get_variable(env_file, 'S3_BUCKET')
S3_DATA_PATH = get_variable(env_file, 'S3_DATA_PATH')
PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = int(get_variable(env_file, 'CHUNKSIZE'))
Q = literal_eval(get_variable(env_file, 'Q'))
Q_TYPE1 = literal_eval(get_variable(env_file, 'Q_TYPE1'))

#First order question types
n_types = 25
question_types1 = {'who': 1, 'whos': 2, 'whose': 3,
                   'what': 4, 'whats': 5,
                   'where': 6, 'wheres': 7,
                   'when': 8, 'whens': 9,
                   'why': 10, 'which': 11, 'is': 12,
                   'can': 13, 'could': 14,
                   'do': 15, 'does': 16,
                   'did': 17, 'will': 18, 'would': 19, 'should': 20,
                   'has': 21, 'have': 22, 'was': 23, 'how': 24}
#***Remember that 0 is the default.

def get_question(s):
    '''Extract first sentence ending in "?",
    We return a string.'''
    if '?' in s:
        sents = nltk.sent_tokenize(s)
        for si in sents:
            if '?' in sents:
                return si
    else:
        return s

def question_type1(q):
    '''Attempt to determine the first order question type, q should
    be a question and we will split on spaces.'''
    qtype = [0]*n_types
    q = str(q)
    for word in q.split():
        word = word.lower()
        word = word.split("'")[0]
        if word in question_types1:
            qtype[question_types1[word]] = 1
    if sum(qtype) == 0:
        qtype[0] = 1
    return list(qtype)

def question_type_chunk(D):
    '''Attempt to classify the type of question'''
    if len(D) > 0:
        D[Q_TYPE1] = D.loc[:, Q].applymap(question_type1)
    return D

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
            D = question_type_chunk(D)
            D.to_csv(PROJECT_DIR + '/data/interim/' + f_name + '_test.csv',
                     index_label = 'id')
    else: #test != 'True'
        pool = Client()
        with pool[:].sync_imports():
            import nltk
        push_res = pool[:].push({'Q' : Q,
                                 'Q_TYPE1' : Q_TYPE1,
                                 'question_type_chunk' : question_type_chunk,
                                 'question_types1' : question_types1,
                                 'get_question' : get_question,
                                 'question_type1' : question_type1,
                                 'n_types' : n_types})
        N_JOBS = len(pool)
        left_indices = range(0, CHUNKSIZE, CHUNKSIZE // N_JOBS)
        right_indices = range(CHUNKSIZE // N_JOBS, CHUNKSIZE + 1,
                              CHUNKSIZE // N_JOBS)

        for f_name in ['train', 'test']:
            D_it = pd.read_csv(PROJECT_DIR + '/data/interim/' + f_name + '.csv',
                               chunksize = CHUNKSIZE, index_col = 'id')
            D0 = D_it.get_chunk()
            D0 = question_type_chunk(D0)
            D0.to_csv(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv',
                      mode = 'w', index_label = 'id')
            del D0

            i = 0
            for Di in D_it:
                i += 1
                if i_max != 0 and i > i_max:
                    break
                print('Classifying question type ',
                      f_name, ' chunk: ', i, end = '\r')
                sys.stdout.flush()
                results = []
                for pi, li, ri in zip(pool, left_indices, right_indices):
                    results.append(pi.apply_async(question_type_chunk,
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
