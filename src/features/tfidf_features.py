import time

import numpy as np

from feature_extractors import skipgram_analyzer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from src.conf import RAW_DATA_FILES, TFIDF_FEATURES, SKIPGRAM_LIST,\
    INTERIM_DATA_DIR, N_NMF_COMPONENTS, PROCESSED_DATA_DIR


def main():
    for f_name in RAW_DATA_FILES:
        # tfidf
        all_questions = np.load(INTERIM_DATA_DIR + 'q_' + f_name + '.npy')
        nrows = len(all_questions) / 2
        assert nrows == int(nrows)
        nrows = int(nrows)

        t = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=TFIDF_FEATURES,
            analyzer=lambda s:
                skipgram_analyzer(s, skipgram_list=SKIPGRAM_LIST)
            )

        print('VECTORIZING', f_name, '...')
        t0 = time.clock()
        tfidf = t.fit_transform(all_questions)
        print('Time:', time.clock() - t0)

        nmf_tfidf = NMF(n_components=N_NMF_COMPONENTS,
                        init='nndsvda')
        print('NMF tfidf', f_name, '...')
        t0 = time.clock()
        W = nmf_tfidf.fit_transform(tfidf)
        print('Time:', time.clock() - t0)

        W = np.abs(W[:nrows, :] - W[nrows:, :])

        X = np.load(INTERIM_DATA_DIR + 'X_' + f_name + '.npy')
        X = np.hstack((X, W))
        X.dump(PROCESSED_DATA_DIR + 'X_' + f_name + '.npy')
    return


if __name__ == '__main__':
    main()
