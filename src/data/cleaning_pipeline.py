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

from ipyparallel import Client, DirectView
from ast import literal_eval

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

import src.data.cleaning_functions
from src.data.cleaning_functions import ExtractCols, ColumnStacker, Splitter,\
    CharFilter, WordTokenize, POSTag, Lemmatize

from src.conf import CHUNKSIZE, valid_chars, INTERIM_HDF_PATH, RAW_DATA_FILES,\
    RAW_DATA_DIR, RAW_DATA_EXT


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
@click.argument('test', type=click.Path(), default='')
def main(test):
    # --------cleaning---------
    get_questions = ExtractCols(['question1', 'question2'])
    # CAN'T PICKLE THE LOGGER OBJECT!  log = True causes a crash

    char_filter = CharFilter(valid_chars=valid_chars, log=False)
    word_tokenizer = WordTokenize()
    pos_tagger = POSTag()
    lemmatizer = Lemmatize()
    cleaning_pipe = Pipeline([('get_questions', get_questions), (
        'char_filter', char_filter), ('word_tokenizer', word_tokenizer), (
            'pos_tagger', pos_tagger), ('lemmatizer', lemmatizer)])

    pool = Client()
    with pool[:].sync_imports():
        import pandas as pd
        import sys
        import os

    pool[:].execute("sys.path.append(os.getcwd())")

    push_res = pool[:].push({'cleaning_pipe': cleaning_pipe})
    N_jobs = len(pool)
    left_indices = range(0, CHUNKSIZE, CHUNKSIZE // N_jobs)
    right_indices = range(CHUNKSIZE // N_jobs, CHUNKSIZE + 1,
                          CHUNKSIZE // N_jobs)

    if test != 'True':
        h5f = tables.open_file(INTERIM_HDF_PATH, mode='w')
        h5f.close()
    for f_name in RAW_DATA_FILES:
        # There is probably a better way to set this up
        raw_data_path = RAW_DATA_DIR + f_name + RAW_DATA_EXT
        D_it = pd.read_csv(raw_data_path, chunksize=CHUNKSIZE, index_col='id')
        n_chunks = count_chunks(D_it)
        if test != 'True':
            h5f = tables.open_file(INTERIM_HDF_PATH, mode='a')
            h5f.create_group('/', f_name)
            h5f.set_node_attr('/' + f_name, 'n_chunks', n_chunks)
            h5f.set_node_attr('/' + f_name, 'chunk_size', CHUNKSIZE)
            h5f.close()

        D_it = pd.read_csv(
            raw_data_path,
            chunksize=CHUNKSIZE,
            index_col='id')

        for i, Di in enumerate(D_it):
            print(f_name, 'chunk', i + 1, '/', n_chunks, end='\r')
            sys.stdout.flush()
            results = []
            for pi, li, ri in zip(pool, left_indices, right_indices):
                results.append(pi.apply_async(cleaning_pipe.transform,
                                              Di[li:ri]))

            for res in results:
                Xi = res.get()
                Di = pd.DataFrame(Xi, columns=['question1', 'question2'])
                try:
                    DX = DX.append(Di)
                except NameError:
                    DX = Di

            # Each Di will only be a part of the whole chunk,
            # but DX should contain all of it.
            DX.to_hdf(
                INTERIM_HDF_PATH,
                key='/' + f_name + '/' + f_name + str(i))
            del DX
        print()
    # char_filter.write_log()

    return


if __name__ == '__main__':
    logging.basicConfig(
        filename='cleaning.log', filemode='w', level=logging.INFO)
    main()
