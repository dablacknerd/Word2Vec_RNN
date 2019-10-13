import spacy
import numpy as np
nlp = spacy.load('en_core_web_lg')

embedding_len = len(nlp.vocab['apple'].vector)

def embedding_matrix(corpus_feature_space_len,feature_space_index):
    # corpus_feature_space_len: Number of distinct words in your dataset
    # embedding_len: Length of each vector representing a word in the word2vec embedding
    # feature_space_index: the list of unique numbers corresponding to the words in your corpus_feature_space
    row_size = corpus_feature_space_len + 1
    column_size = embedding_len
    embedding_matrix = np.zeros((row_size, column_size))
    for word, idx in feature_space_index.items():
        try:
            word_embedding = nlp.vocab[word].vector
            embedding_matrix[idx] = word_embedding
        except:
            pass
    return embedding_matrix
