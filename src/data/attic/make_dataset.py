# -*- coding: utf-8 -*-
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

S3_BUCKET = get_variable(env_file, 'S3_BUCKET')
S3_DATA_PATH = get_variable(env_file, 'S3_DATA_PATH')
PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = 1024

Q = ['question1', 'question2']
Q_word_tokenized = ['question1_word_tokenized', 'question2_word_tokenized']
Q_tag = ['question1_pos_tag', 'question2_pos_tag']

def wtokenize_ptag_chunk(Di):
    '''word tokenize and position tag chunks'''
    if len(Di) > 0:
        Di.loc[:, Q] = Di.loc[:, Q].applymap(str) #Ensure we have strings
        Di[Q_word_tokenized] = Di.loc[:, Q].applymap(nltk.word_tokenize)
        Di[Q_tag] = Di.loc[:, Q_word_tokenized].applymap(nltk.pos_tag)
    return Di

def tokenize_dataset(f_name, ix_id):
    pool = Client()
    with pool[:].sync_imports():
        import nltk
    push_res = pool[:].push({'Q': Q,
                             'Q_word_tokenized': Q_word_tokenized,
                             'Q_tag': Q_tag})
    push_res.get();

    N_JOBS = len(pool)
    left_indices = range(0, CHUNKSIZE, CHUNKSIZE // N_JOBS)
    right_indices = range(CHUNKSIZE // N_JOBS, CHUNKSIZE + 1,
                          CHUNKSIZE // N_JOBS)

    D_it = pd.read_csv('s3://' + S3_BUCKET + '/' +
                       S3_DATA_PATH + '/raw/' + f_name,
                       chunksize = CHUNKSIZE, index_col = ix_id)

    i = 0
    print('Processing ', f_name, ' chunk: ', i, end = '\r')
    sys.stdout.flush()
    D0 = D_it.get_chunk()
    D0 = wtokenize_ptag_chunk(D0)
    D0.to_csv(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv', mode = 'w',
              index_label = 'id')
    del D0
    for Di in D_it:
        i += 1
        print('Processing ', f_name, ' chunk: ', i, end = '\r')
        sys.stdout.flush()
        results = []
        for pi, li, ri in zip(pool, left_indices, right_indices):
            results.append(pi.apply_async(wtokenize_ptag_chunk, Di[li:ri]))

        for res in results:
            Di = res.get()
            if len(Di) > 0:
                Di.to_csv(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv', 
                          mode = 'a', header = False, index_label = 'id')
            del Di

    print()
    s3 = boto3.resource('s3')
    b = s3.Bucket(S3_BUCKET)
    b.upload_file(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv',
                  'quora_question_pairs/data/processed/' + f_name)
    os.remove(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv')
    return

@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():#input_filepath, output_filepath):
    """ Runs data processing scripts """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    for f_name, ix_id in zip(['train.csv', 'test.csv'], ['id', 'test_id']):
        print('Tokenizing dataset', f_name)
        tokenize_dataset(f_name, ix_id)
        logger.info('Finished dataset %s' % f_name)
    return

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    main()
