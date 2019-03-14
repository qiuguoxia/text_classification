# training data
training_data = "./data/training.tsv"
test_data = "./data/test.tsv"

# embeddings
glove = "/home/rs/Documents/bigdata/embeddings/glove/glove.twitter.27B.100d.txt" # pretrained word embeddings for tweets
dims = 100 # dimensions of the word vectors
vocab_size = 10000 # size of the dictionary
sequence_length = 25 # length of input sequences
binary= False # whether the glove file is in binary format

# model
batch_size = 32
verbosity=1
dropout = 0.5
nb_epoch = 100
nb_filter = 250
filter_length = 5

