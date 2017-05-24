'''
This is intended to build the feature matrix X that will be fed into the
classifier pipeline.
'''
import os
import sys
import click
import tables
import numpy as np
import pandas as pd

from ipyparallel import Client

from sklearn.pipeline import Pipeline, FeatureUnion

from feature_extractors import *
from src.data.cleaning_functions import WordTokenize

from src.conf import CHUNKSIZE, INTERIM_HDF_PATH, RAW_DATA_FILES,\
    INTERIM_DATA_DIR, Q, FEATURES_DIR

question_types = {
    'who': 1,
    'whos': 2,
    'whose': 3,
    'what': 4,
    'whats': 5,
    'where': 6,
    'wheres': 7,
    'when': 8,
    'whens': 9,
    'why': 10,
    'which': 11,
    'is': 12,
    'can': 13,
    'could': 14,
    'do': 15,
    'does': 16,
    'did': 17,
    'will': 18,
    'would': 19,
    'should': 20,
    'has': 21,
    'have': 22,
    'was': 23,
    'how': 24
}


@click.command()
@click.argument('num_rows', type=click.Path(), default=-1)
def main(num_rows):
    num_rows = int(num_rows)

    # ---------preprocessing----------
    get_questions = ExtractCols(['question1', 'question2'])

    # ---------question typing---------
    sent_tokenizer = SentTokenize()
    question_typer = QuestionTypes(question_types=question_types)
    question_type_pipe = Pipeline([('sent_tokenizer', sent_tokenizer),
                                   ('question_typer', question_typer)])

    # --------distance calculation-------
    calc_masi_dist = MasiDistance()
    # These have tuning parameters and
    # could go through cv...
    calc_edit_dist1 = EditDistance()
    calc_edit_dist2 = EditDistance(sub_cost=2.0)
    calc_edit_dist3 = EditDistance(sub_cost=0.5)
    calc_edit_dist4 = EditDistance(transpositions=True)
    calc_jacc_dist = JaccardDistance()
    dist_fu = FeatureUnion(
        [('calc_masi_dist', calc_masi_dist),
         ('calc_edit_dist1', calc_edit_dist1),
         ('calc_edit_dist2', calc_edit_dist2),
         ('calc_edit_dist3', calc_edit_dist3),
         ('calc_edit_dist4', calc_edit_dist4),
         ('calc_jacc_dist', calc_jacc_dist)],
        n_jobs=1)

    word_tokenizer = WordTokenize()
    dist_pipe = Pipeline([('word_tokenizer', word_tokenizer),
                          ('dist_fu', dist_fu)])

    output_fu = FeatureUnion(
        [('dist_pipe', dist_pipe),
         ('question_type_pipe', question_type_pipe)],
        n_jobs=1)

    # --------final assembly----------
    data_pipe = Pipeline([('get_questions', get_questions),
                          ('output_fu', output_fu)])  # output feature union

    pool = Client()
    pool[:].map(os.chdir, [FEATURES_DIR]*len(pool))
    with pool[:].sync_imports():
        pass

    pool[:].push({'data_pipe': data_pipe})

    n_jobs = len(pool)
    left_indices = range(0, CHUNKSIZE, CHUNKSIZE // n_jobs)
    right_indices = range(CHUNKSIZE // n_jobs, CHUNKSIZE + 1,
                          CHUNKSIZE // n_jobs)

    for f_name in RAW_DATA_FILES:
        h5f = tables.open_file(INTERIM_HDF_PATH, 'r')
        n_chunks = h5f.get_node_attr('/' + f_name, 'n_chunks')
        h5f.close()
        for i in range(n_chunks):
            print('chunk', i + 1, '/', n_chunks, end='\r')
            sys.stdout.flush()
            Di = pd.read_hdf(INTERIM_HDF_PATH,
                             key='/' + f_name + '/' + f_name + str(i))

            q1_i = Di.loc[:, Q[0]].values
            q2_i = Di.loc[:, Q[1]].values
            try:
                q1 = np.concatenate((q1, q1_i))
                q2 = np.concatenate((q2, q2_i))
            except NameError:
                q1 = q1_i
                q2 = q2_i

            results = []  # Send to workers
            # It would be much faster to load chunks of data off disk
            # as fast as possible and use some kind of switch on the
            # pool to send them chunks when they are ready.  As is,
            # a huge amount of time (probably most) is spent waiting
            # for io.  Not to mention time used copying data back
            # and forth between the processes.

            # Maybe the best approach is for each process to load and
            # store it's own results to the database independently,
            # similar to what I did for the dwglasso application.
            for pi, li, ri in zip(pool, left_indices, right_indices):
                if len(Di[li:ri]) > 0:
                    results.append(pi.apply_async(data_pipe.fit_transform,
                                                  Di[li:ri]))
            for res in results:
                Xi = res.get()
                try:
                    X = np.vstack((X, Xi))
                except NameError:
                    X = Xi

        X.dump(INTERIM_DATA_DIR + 'X_dist_' + f_name + '.npy')
        del X  # Free up memory

        nrows = len(q1)
        assert nrows == len(q2), 'q1 and q2 not equal length!'

        q1 = np.append(q1, q2)
        del q2
        q1.dump(INTERIM_DATA_DIR + 'q_' + f_name + '.npy')
        del q1
    return


if __name__ == '__main__':
    main()
