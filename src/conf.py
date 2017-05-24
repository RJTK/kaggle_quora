import os
cwd = os.getcwd()

# Directories for importing on ipyparallel
FEATURES_DIR = cwd + '/src/features'

# Data file information
RAW_DATA_DIR = cwd + '/data/raw/'
RAW_DATA_FILES = ['test', 'train']
TRAIN_FILE = 'train'
RAW_DATA_EXT = '.csv'
INTERIM_HDF_PATH = cwd + '/data/interim/interim_data.hdf'
INTERIM_DATA_DIR = cwd + '/data/interim/'
PROCESSED_DATA_DIR = cwd + '/data/processed/'

# The character set used to filter all of the questions
valid_chars = set(
    '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?.!:;, ')

# s3://rjtk/...
S3_BUCKET = 'rjtk'
S3_DATA_PATH = 'quora_question_pairs/dara'

# Number of rows of data read at a time
CHUNKSIZE = 2**14  # Chunk size is split between each ipyparallel worker
TEST_ROWS = 16
MASI_DISTANCE = 'masi_dist'
EDIT_DISTANCE = 'edit_dist'
JACCARD_DISTANCE = 'jaccard_dist'
Q = ['question1', 'question2']
Q_WORD_TOKENIZED = ['question1_word_tokenized',
                    'question2_word_tokenized']
Q_TAGGED = ['question1_pos_tagged',
            'question2_pos_tagged']
Q_TYPE = ['question1_type1',
          'question2_type2']

# tfidf/nmf params
TFIDF_FEATURES = 20000
N_NMF_COMPONENTS = 75
# SKIPGRAM_LIST = [(1, 0), (2, 3), (3, 2)]
SKIPGRAM_LIST = [(1, 0)]
