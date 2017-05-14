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
from nltk.corpus import wordnet

S3_BUCKET = get_variable(env_file, 'S3_BUCKET')
S3_DATA_PATH = get_variable(env_file, 'S3_DATA_PATH')
PROJECT_DIR = get_variable(env_file, 'PROJECT_DIR')
CHUNKSIZE = int(get_variable(env_file, 'CHUNKSIZE'))
Q_WORD_TOKENIZED = literal_eval(get_variable(env_file, 'Q_WORD_TOKENIZED'))
Q_TAGGED = literal_eval(get_variable(env_file, 'Q_TAGGED'))

def get_wordnet_pos(treebank_tag):
    '''
    Converts the treebank POS tag to
    a tag compaitible with the lemmatizer
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#This is pretty bad style, but it makes sense with my workflow
if __name__ == '__main__':
    wnl = nltk.stem.WordNetLemmatizer()
def lemmatize_pos_list(lst_str):
    '''
    -Lemmatizes a list of (str, pos) pairs
    -The list is provided as a string literal (from pandas df)
    -Assumes a lemmatizer "wnl" is in the namespace
    '''
    return [wnl.lemmatize(ti[0], get_wordnet_pos(ti[1])) 
            for ti in literal_eval(lst_str)]

def lemmatize_chunk(D):
    '''Lemmatize the data from position tagged tuples'''
    if len(D) > 0:
        s = D.loc[:, Q_TAGGED].applymap(lemmatize_pos_list)
        for ql, qr in zip(Q_WORD_TOKENIZED, Q_TAGGED):
            D.loc[:, ql] = s[qr]
    return D

@click.command()
@click.argument('test', type = click.Path(), default = 'False')
@click.argument('i_max', type = click.Path(), default = 0)
def main(test, i_max):
    i_max = int(i_max)
    if test == 'True': #Don't chunk
        for f_name in ['train', 'test']:
            print('Lemmatizing (test)', f_name)
            D = pd.read_csv(PROJECT_DIR + '/data/interim/' + f_name + '_test.csv',
                            index_col = 'id')
            D = lemmatize_chunk(D)
            D.to_csv(PROJECT_DIR + '/data/interim/' + f_name + '_test.csv',
                     index_label = 'id')
    else: #test != 'True'
        pool = Client()
        with pool[:].sync_imports():
            import nltk
            from nltk.corpus import wordnet
            from ast import literal_eval
        push_res = pool[:].push({'Q_TAGGED' : Q_TAGGED,
                                 'Q_WORD_TOKENIZED' : Q_WORD_TOKENIZED,
                                 'wnl' : wnl,
                                 'lemmatize_pos_list' : lemmatize_pos_list,
                                 'get_wordnet_pos' : get_wordnet_pos})
        N_JOBS = len(pool)
        left_indices = range(0, CHUNKSIZE, CHUNKSIZE // N_JOBS)
        right_indices = range(CHUNKSIZE // N_JOBS, CHUNKSIZE + 1,
                              CHUNKSIZE // N_JOBS)

        for f_name in ['train', 'test']:
            D_it = pd.read_csv(PROJECT_DIR + '/data/interim/' + f_name + '.csv',
                               chunksize = CHUNKSIZE, index_col = 'id')
            D0 = D_it.get_chunk()
            D0 = lemmatize_chunk(D0)
            D0.to_csv(PROJECT_DIR + '/data/interim/' + 'D_tmp.csv',
                      mode = 'w', index_label = 'id')
            del D0

            i = 0
            for Di in D_it:
                i += 1
                if i_max != 0 and i > i_max:
                    break
                print('Lemmatizing ', f_name, ' chunk: ', i, end = '\r')
                sys.stdout.flush()
                results = []
                for pi, li, ri in zip(pool, left_indices, right_indices):
                    results.append(pi.apply_async(lemmatize_chunk, Di[li:ri]))
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
