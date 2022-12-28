import random
import torch
import numpy as np

from settings import ORIGIN_SEQ, WORD_LENGTH, WORD2VEC_MODEL_PATH, VECTOR_SIZE
from word2vec import ProtWord2Vec

# In this script, a directed evolution experiment is simulated, and 
# sequences are represented using the word2vec algorithm (using the 
# class ProtWord2Vec in the word2vec.py script).
# Contains generators to reduce memory consumption during the fetching 
# of sequences in the training and test (each iteration creates the 
# representation of the batch of sequences using word2vec).

class Experiment:
    """
    Create Artificial experiment of protein engineering.
    Create one "origin" sequence, and then in each iteration insert cumulative mutations.
    The Protein efficiency increases with each iteration.
    """

    def __init__(self, num_iter, efficiency_interval, num_mutations):
        """
        :param num_iter:            number of the iteration in the "experiment"
        :param efficiency_interval: The amount by which efficiency increases with each iteration
        :param num_mutations:       number of mutations per iteration
        """
        self.num_iter = num_iter
        self.efficiency_interval = efficiency_interval
        self.num_mutations = num_mutations
        # The "labels" of the sequences (in the regression problem) - the efficiencies of the sequences
        self.efficiencies = [1 + i * self.efficiency_interval for i in range(self.num_iter + 1)]

    def create_exp(self, origin_seq):
        """
        Create sequences from the experiments

        :param origin_seq: the origin sequence

        :return:           list of mutated sequences and list of their efficiencies.
        """
        aa_list = list('ARNDCEQGHILKMFPSTWYV')
        sequences = [origin_seq]
        for i in range(self.num_iter):
            mut_locs = random.sample(range(len(origin_seq)), self.num_mutations)
            mutations = random.sample(aa_list, self.num_mutations)
            sequence = list(sequences[-1])
            for loc, mutation in zip(mut_locs, mutations):
                sequence[loc] = mutation
            sequences.append(''.join(sequence))
        return sequences

    def seq_to_words(self, sequences, word_length):
        """
        Divide each sequence of AA's to words of lenght "word_length"

        :param sequences:    list of sequences of AA's
        :param word_length:  length of word

        :return: list of lists, each internal list is words of one protein sequence
        """
        seq_words = []
        for i, seq in enumerate(sequences):
            seq_words.append([seq[i:i + word_length] for i in range(len(seq) - word_length + 1)])
        return seq_words

    def words_to_vector(self, seq_words, model_path, word_length, sum_vectors=False):
        """
        This function is an generator.
        Vectors that represent the "words" of the protein (AA) (using word2vec 
        algorithm).
        Can return numpy matrix (the vectors of all words) for each sequence, or one 
        vector - the sum of the vectors of all words (depends on the sum_vectors argument).
        This function is an generator, yield representation of one protein and its 
        efficiency in each calling.

        :param seq_words:     list of lists, each internal list is words of one protein 
                              sequence.
        :param model_path:    path to word2vec model
        :param word_length:   length of "words" - sequences of (overlapped) amino acids
        :param sum_vectors:   return sum of all vectors (for classical models) 
                              or matrix (for CNN).

        :yield:               tuple: (numpy array - a vector represent the words (sum of the vectors of the words),
                                     efficiency of the protein)
                        or    tuple: (numpy matrix - each row is vector represents one word,
                                     efficiency of the protein)
        """
        
        prot_w2v = ProtWord2Vec(model_path, word_length)
        word2vec_model = prot_w2v.load_trained_model()
        for seq, eff in zip(seq_words, self.efficiencies):
            if sum_vectors:
                yield prot_w2v.get_word_vector(word2vec_model, seq).sum(axis=0), eff
            else:
                yield prot_w2v.get_word_vector(word2vec_model, seq), eff

    def get_batch(self, seq_words, batch_size, channels, width, shuffle=False):
        """
        This function is an generator.
        It get batches of images (represented as vectors)
        
        :param seq_words:  all sequence of proteins (represented as words).
        :param batch_size:      batch size (number of proteins)
        :param channels:   length of the vector representation.
        :param width:      nunmber of words in proteing (the length of the "image").

        :return: batch of images (batch, channels, length), and labels
        """
        images = np.zeros((batch_size, channels, width))  # batch, channel (vector length), word number
        effs = np.zeros(batch_size)
        j = 0
        if shuffle:
            random.shuffle(seq_words)

        # In each iteration sends one protein sequence (list of words) from the batch to word2vec to get matrix -
        # each row represents one word
        for i, (vec, eff) in enumerate(self.words_to_vector(seq_words, WORD2VEC_MODEL_PATH, WORD_LENGTH)):
            images[j] = vec.T
            effs[j] = eff
            j += 1
            if (i + 1) % batch_size == 0:
                yield torch.from_numpy(images).float(), torch.from_numpy(effs).float()
                images = np.zeros((batch_size, channels, width))  # batch, channel (vector length), word number
                effs = np.zeros(batch_size)
                j = 0

    def calculate_batch_norm(self, seq_words, channels, width):
        """
        Calculate batch normalization - mean and std among the batches for each 
        channel in separate.
        #https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8

        :param seq_words:  all sequence of proteins (represented as words).
        :param batch:      batch size (number of proteins)
        :param channels:   length of the vector representation.
        :param width:      nunmber of words in proteing (the length of the "image").

        :return: mean and std of each channel - one number per channel
        """
        # The batch size is all sequences
        images, _ = next(
            self.get_batch(seq_words, len(seq_words), channels, width, shuffle=False))  # call to all sequences at once
        mean_batch = images.mean(axis=(0, 2))
        std_batch = images.std(axis=(0, 2))
        return mean_batch, std_batch


if __name__ == '__main__':
    exp = Experiment(num_iter=500, efficiency_interval=1, num_mutations=1)
    sequences = exp.create_exp(ORIGIN_SEQ)
    seq_words = exp.seq_to_words(sequences, word_length=WORD_LENGTH)
    mean_channels, std_channels = exp.calculate_batch_norm(seq_words, VECTOR_SIZE, len(seq_words[0]))
    # seq_vectors = exp.words_to_vector(seq_words, WORD2VEC_MODEL_PATH, WORD_LENGTH, False)
    # for i in range(1):
    #     print(next(seq_vectors)[0].shape)
    for vec, eff in exp.words_to_vector(seq_words, WORD2VEC_MODEL_PATH, WORD_LENGTH):
        print(vec, eff)

    # print(vector_words.shape)
