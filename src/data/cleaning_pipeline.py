'''
This script reads in the raw data and produces lemmatized questions with only
a strict set of valid characters.

Input: Raw data files

Output: hdf5 database organized as follows:
  #The training data
  /train.n_chunks = n_chunks <-- Number of pandas df chunks
  /train.chunk_size = chunk_size <-- The size of each chunk
  /train/train<i> (i = 1..n_chunks) <-- The pandas df

  #The test data
  /test.n_chunks = n_chunks <-- Number of pandas df chunks
  /test.chunk_size = chunk_size <-- The size of each chunk
  /test/test<i> (i = 1..n_chunks) <-- The pandas df

Note that this doesn't keep track of the labels, they would need to be loaded
in later when the model is trained.
'''

import os
import sys
import click
import logging
import nltk
import h5py
import logging
import tables

import numpy as np
import pandas as pd

from dotenv import get_variable; env_file = '/home/ubuntu/science/quora_question_pairs/.env'
from ipyparallel import Client, DirectView
from ast import literal_eval

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

import cleaning_functions
from cleaning_functions import ExtractCols, ColumnStacker, Splitter,\
    CharFilter, WordTokenize, POSTag, Lemmatize

PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = int(get_variable(env_file, 'CHUNKSIZE'))

valid_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?.!:;, ')

def count_rows(D_it):
    '''
    Streams an iterable dataframe through and counts the number of rows.
    '''
    num_rows = 0
    for Di in D_it:
        num_rows += len(Di)
    return num_rows

def count_chunks(D_it):
    '''
    Streams an iterable dataframe through and counts the number iterates.
    '''
    i = 0
    for Di in D_it:
        i += 1
    return i

@click.command()
@click.argument('f_name', type = click.Path())
@click.argument('test', type = click.Path(), default = '')
def main(f_name, test):
    #--------cleaning---------
    get_questions = ExtractCols(['question1', 'question2'])
    #CAN'T PICKLE THE LOGGER OBJECT!  log = True causes a crash
    char_filter = CharFilter(valid_chars = valid_chars,
                             log = False)
    word_tokenizer = WordTokenize()
    pos_tagger = POSTag()
    lemmatizer = Lemmatize()
    cleaning_pipe = Pipeline([('get_questions', get_questions),
                              ('char_filter', char_filter),
                              ('word_tokenizer', word_tokenizer),
                              ('pos_tagger', pos_tagger),
                              ('lemmatizer', lemmatizer)])

    pool = Client()
    with pool[:].sync_imports():
        import pandas as pd
        import sys, os

    pool[:].execute("sys.path.append(os.getcwd())")

    push_res = pool[:].push({'cleaning_pipe': cleaning_pipe})
    N_jobs = len(pool)
    left_indices = range(0, CHUNKSIZE, CHUNKSIZE // N_jobs)
    right_indices = range(CHUNKSIZE // N_jobs, CHUNKSIZE + 1,
                          CHUNKSIZE // N_jobs)

    #There is probably a better way to set this up
    D_it = pd.read_csv(PROJECT_DIR + '/data/raw/' + f_name + '.csv',
                       chunksize = CHUNKSIZE, index_col = 'id')
    n_chunks = count_chunks(D_it)
    if test != 'True':
        h5f = tables.open_file(PROJECT_DIR + '/data/interim/interim_data.hdf',
                               mode = 'w')
        h5f.create_group('/', f_name)
        h5f.set_node_attr('/' + f_name, 'n_chunks', n_chunks)
        h5f.set_node_attr('/' + f_name, 'chunk_size', CHUNKSIZE)
        h5f.close()

    D_it = pd.read_csv(PROJECT_DIR + '/data/raw/' + f_name + '.csv',
                       chunksize = CHUNKSIZE, index_col = 'id')

    for i, Di in enumerate(D_it):
        print('chunk ', i + 1, '/', n_chunks, end = '\r')
        sys.stdout.flush()
        results = []
        for pi, li, ri in zip(pool, left_indices, right_indices):
            results.append(
                pi.apply_async(cleaning_pipe.transform, Di[li:ri]))

        for res in results:
            Xi = res.get()
            Di = pd.DataFrame(Xi, columns = ['question1', 'question2'])
            if test == 'True':
                print(Di)
                break
            Di.to_hdf(PROJECT_DIR + '/data/interim/interim_data.hdf',
                      key = '/' + f_name + '/' + f_name + str(i))
    print()
    #char_filter.write_log()

    return

if __name__ == '__main__':
    logging.basicConfig(filename = 'cleaning.log', filemode = 'w',
                        level = logging.INFO)
    main()
