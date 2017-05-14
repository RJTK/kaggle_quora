# -*- coding: utf-8 -*-
import os
import sys
import click
import logging

import numpy as np
import pandas as pd

import boto3

from dotenv import get_variable; env_file = '/home/ubuntu/science/quora_question_pairs/.env'

S3_BUCKET = get_variable(env_file, 'S3_BUCKET')
S3_DATA_PATH = get_variable(env_file, 'S3_DATA_PATH')
PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = 4*int(get_variable(env_file, 'CHUNKSIZE'))
TEST_ROWS = int(get_variable(env_file, 'TEST_ROWS'))

@click.command()
@click.argument('test', type = click.Path(), default = 'False')
def main(test):
    if test == 'True': #Don't chunk
        for f_name, ix_id in zip(['train', 'test'], ['id', 'test_id']):
            print('Downloading (test)', f_name)
            D = pd.read_csv('s3://' + S3_BUCKET + '/' +
                            S3_DATA_PATH + '/raw/' + f_name + '.csv',
                            index_col = ix_id, nrows = TEST_ROWS)
            D.to_csv(PROJECT_DIR + '/data/raw/' + f_name + '_test.csv',
                     mode = 'w', index_label = 'id')
    else:
        for f_name, ix_id in zip(['train', 'test'], ['id', 'test_id']):        
            D_it = pd.read_csv('s3://' + S3_BUCKET + '/' +
                               S3_DATA_PATH + '/raw/' + f_name + '.csv',
                               chunksize = CHUNKSIZE, index_col = ix_id)
            D0 = D_it.get_chunk()
            D0.to_csv(PROJECT_DIR + '/data/raw/' + f_name + '.csv',
                      mode = 'w', index_label = 'id')
            del D0

            i = 0
            for Di in D_it:
                i += 1
                print('Downloading ', f_name, ' chunk: ', i, end = '\r')
                sys.stdout.flush()
                Di.to_csv(PROJECT_DIR + '/data/raw/' + f_name + '.csv',
                          mode = 'a', header = False, index_label = 'id')
            print()
    return

if __name__ == '__main__':
  main()
