import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
# # To correct spelling
# from autocorrect import spell
# # To create the bag of words
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.manifold import TSNE
# # Plots
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pickle
# import os


def get_texts_from_file(file_path, column_name):
    """Read text from 'file_path' and returns the content in 'column_name' as a 
       list of strings.
    """
    data = pd.read_csv(file_path)
    texts = data[column_name].values.tolist()
    return texts


def tokenize_texts(texts):
    """Remove invalid characters and tokenize texts."""
    texts = [re.sub('[^A-Za-zçáéíóúàããẽõ]', ' ', text).lower()
             for text in texts]
    texts = [word_tokenize(text, language='portuguese') for text in texts]
    return texts


def remove_stopwords(texts):
    """Remove stopwords from the lists of tokens"""
    stopwords_ = stopwords.words('portuguese')
    for text in texts:
        for word in text:
            if word in stopwords_:
                text.remove(word)
    return texts


def stemmize_text(texts):
    """Text stemmize each token in the list of tokens"""
    # Stemming and correct spelling
    stemmer = RSLPStemmer()
    texts_ = []
    for text in texts:
        text_ = [stemmer.stem(w) for w in text]
        texts_.append(text_)
    return texts_


def create_corpus(texts):
    """Join processed list of tokens"""
    corpus = [' '.join(text) for text in texts]
    return corpus


# Podemos alterar essa função para receber valores de corte, ou utilizar todos
# os termos
def get_bag_of_words(corpus, max_features=1000):
    """Create Bag-of-Words representation of the corpus"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus).toarray()

    return X


# Descomentar para testar as funcionalidades
# if __name__ == '__main__':
#     texts = get_texts_from_file('../data/febre_amarela_jun17-out18.csv',
#                                 'title')
#     texts_tokenized = tokenize_texts(texts)
#     texts_stopwords_rem = remove_stopwords(texts_tokenized)
#     texts_stemmized = stemmize_text(texts_stopwords_rem)
#     corpus = create_corpus(texts_stemmized)
#     bow = get_bag_of_words(corpus)
#     print(bow[0])
