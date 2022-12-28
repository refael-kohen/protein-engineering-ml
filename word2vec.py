# This script create word2vec network using Gensim package for "words" of proteins.
# "word" is sequence of amino acids (AA) of length WORD_LENGTH.

# The human proteins are downlaoded from uniProt (80,581 proteins):
# https://www.uniprot.org/proteomes/UP000005640

# Manual for train word2vec network with gensim on triplets of amino-acids:
# https://radimrehurek.com/gensim/models/word2vec.html

from gensim.models import Word2Vec

from settings import WORD_LENGTH, WORD2VEC_MODEL_PATH, PROT_SEQ_FILE, VECTOR_SIZE


class ProtWord2Vec:
    """
    Train word2vec model of gensim package on protein sequences
    """

    def __init__(self, model_path, word_length):
        """
        :param model_path:    path to word2vec model
        :word_length:         length of "words" - sequences of (overlapped) amino acids
        """
        self.model_path = model_path
        self.word_length = word_length

    def load_aa_words(self, prot_seq_file):
        """
        Read sequences from fasta file and divide the sequencs to words (sequence of AA of length WORD_LENGTH)

        :param prot_seq_file: path to fasta file with 80,581 sequences of human proteins
        :return list of lists - each line is one protein divided to words
        """
        sentences = []
        seq_num = -1

        with open(prot_seq_file, 'r') as fh:
            for line in fh:
                if line.startswith('>'):
                    seq_num += 1
                    sentences.append([])
                    continue
                line = line.rstrip()
                sentences[seq_num] += [line[i:i + self.word_length] for i in range(len(line) - self.word_length + 1)]
        return sentences

    def train_word2vec(self, prot_seq_file, vector_size):
        """
        Train word2vec model with the sentences (sequence of words - each word is sequence of AA of length WORD_LENGTH).
        Save the model on the disk.

        :param prot_seq_file:  path to fasta file with 80,581 sequences of human proteins
        :param vector_size:    length of the vector that represent the word

        """
        sentences = self.load_aa_words(prot_seq_file)
        model = Word2Vec(sentences=sentences, size=vector_size, window=5, min_count=1, workers=4)
        model.save(self.model_path)

    def load_trained_model(self):
        """
        Load the trained mode from the disk
        :return: the trained model
        """
        word2vec_model = Word2Vec.load(self.model_path)
        # For word_length=3 the output (vocabulary size) is 8498, but the maximum
        # words number can be up to 20^3=8000. It is because there is variations of AA, 
        # so we have more than 20 letters.
        # print(word2vec_model.wv.vectors.shape(0))  # number words in the vocabulary
        return word2vec_model

    def get_word_vector(self, word2vec_model, words):
        """
        Get the vector represents the given word.

        :param word2vec_model:  instance of the model
        :param words:           the input word or list of words

        :return: vector (numpy array) represents the input word or numpy matrix (row for each word)
        """
        vector = word2vec_model.wv[words]
        # print(word2vec_model.wv.most_similar(word, topn=10))
        return vector


if __name__ == '__main__':
    prot_w2v = ProtWord2Vec(WORD2VEC_MODEL_PATH, WORD_LENGTH, VECTOR_SIZE)
    # prot_w2v.train_word2vec(PROT_SEQ_FILE, VECTOR_SIZE)
    word2vec_model = prot_w2v.load_trained_model()
    word_vector = prot_w2v.get_word_vector(word2vec_model, 'AAA')
