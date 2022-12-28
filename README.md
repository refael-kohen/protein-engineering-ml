Implementation of the paper "Deep Dive into Machine Learning Models for Protein Engineering".

- Simulation of a directed evolution experiment (improving protein properties). The experiment involves inserting new mutations in each iteration. The output sequences are all the same length.
- Represent the protein sequences with word2vec: train word2vec (Gensim package) on triplets of amino acids.
- Building a 1d CNN network (regression loss function) where the sequence of the triplets of amino acids (1d vector) is the width and the vectors from word2vec are the channels.