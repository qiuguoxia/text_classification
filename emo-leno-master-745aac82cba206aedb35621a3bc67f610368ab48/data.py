from random import Random

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

import config
import embeddings
import twitter_tokenizer


def read_data(path, threshold=.5):
    """
    Load EmoInt data into lists.
    """
    sentences, values = [], []
    with open(path, 'r') as file:
        for line in file.readlines():
            _, sent, cat, val = line.split('\t')
            if float(val.strip()) > threshold:  # map onto categories
                sentences.append(sent)
                values.append(cat)
        return sentences, values


def load_X_y(path):
    """
    Get input and target points from EmoInt file.
    :param path: EmoInt tsv file
    :return: Input and target points.
    """
    encoder = LabelBinarizer()
    encoder.fit(['anger', 'fear', 'sadness', 'joy'])
    X, y = read_data(path)
    y = encoder.transform(y)
    X = [twitter_tokenizer.tokenize(x) for x in X]
    Random(42).shuffle(X)
    Random(42).shuffle(y)
    return X, y


def training(cfg):
    """
    Get training data from EmoInt file.
    :param cfg: Config file w/ key training_data set.
    :return: Input and target points.
    """
    return load_X_y(cfg.training_data)


def test(cfg):
    """
    Get test data from EmoInt file.
    :param cfg: Config file w/ key training_data set.
    :return: Input and target points.
    """
    return load_X_y(cfg.test_data)


def load_data(cfg):
    """
    Get weights matrix for embeddings, training and test data.
    :param cfg: Config file w/ training and test file paths as well as sequence length keys set.
    :return: Weights matrix, X,y for training, X,y for testing.
    """
    X_train, y_train = training(cfg)
    X_test, y_test = test(cfg)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train + X_test)
    X_train_ints = tokenizer.texts_to_sequences(X_train)
    X_test_ints = tokenizer.texts_to_sequences(X_test)
    X_train_ints = sequence.pad_sequences(X_train_ints, maxlen=config.sequence_length, padding='post')
    X_test_ints = sequence.pad_sequences(X_test_ints, maxlen=config.sequence_length, padding='post')
    lookup = tokenizer.word_index
    weights = embeddings.weights(lookup)
    return weights, X_train_ints, y_train, X_test_ints, y_test
