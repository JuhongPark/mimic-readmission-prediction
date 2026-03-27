import os
from gensim.models import KeyedVectors

BIOWORDVEC_PATH = os.environ.get("BIOWORDVEC_PATH", "")


def load_biowordvec(path=None):
    if path is None:
        path = BIOWORDVEC_PATH
    return KeyedVectors.load_word2vec_format(path, binary=True)


if __name__ == '__main__':
    model = load_biowordvec()
    vector = model.get_vector('influenza')
    print(vector)
