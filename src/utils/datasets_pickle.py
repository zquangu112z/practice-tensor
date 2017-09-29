import pickle
from .log import log
import timeit

# change dir for your convenient
TRAIN_VEC_DIR = 'data/train_vecs.pkl'
TRAIN_LABEL_DIR = 'data/train_labels.pkl'
TEST_VEC_DIR = 'data/test_vecs.pkl'
TEST_LABEL_DIR = 'data/test_labels.pkl'


def get_data_sets_pickle():
    start1 = timeit.default_timer()
    try:
        log(r'Loading dataset using pickle')

        with open(TRAIN_VEC_DIR, 'rb') as f:
            train_line_vectors = pickle.load(f)
        with open(TEST_VEC_DIR, 'rb') as f:
            test_line_vectors = pickle.load(f)
        with open(TRAIN_LABEL_DIR, 'rb') as f:
            train_line_labels = pickle.load(f)
        with open(TEST_LABEL_DIR, 'rb') as f:
            test_line_labels = pickle.load(f)

        stop1 = timeit.default_timer()
        log('Load dataset using pickle successful!:',
              stop1 - start1, 'seconds')
        return train_line_vectors, test_line_vectors, train_line_labels, test_line_labels
    except:
        log('No existing dataset pickle')
