import numpy as np

import config


def vocab():
    """
    Returns a dictionary that maps words onto their embeddings.
    :return: lookup for word embeddings
    """
    f = open(config.glove)
    vocab = {}
    vocab_count = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        vocab[word] = coefs
        vocab_count += 1
        if vocab_count >= config.vocab_size:
            break
    f.close()
    return vocab


def weights(lookup):
    """
    Returns weights for embedding matrix.
    :param lookup: list of tuples, mapping words onto their rank in a corpus, e.g. [(the, 0), (is, 1),...]
    :param vocab: lookup for word embeddings
    :return: embeddings matrix
    """
    embedding_matrix = np.zeros((len(lookup) + 1, config.dims))
    vc = vocab()
    for word, index in lookup.items():
        embedding_vector = vc.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector
    return embedding_matrix
