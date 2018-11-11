import pandas as pd
import re
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_texts_from_file(file_path, column_name):
    """Read texts from 'file_path' and returns the content in 'column_name' as
       a list of strings.
    """
    data = pd.read_json(file_path)
    texts = data[column_name].values.tolist()
    return texts


def remove_duplicated_texts(texts):
    """Remove duplicated text entries."""
    seen = set()
    texts_ = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            texts_.append(text)
    return texts_


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
    """Stemmize each token in the list of tokens"""
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

def get_topics(corpus,n_topics):
    dictionary = Dictionary(corpus)
    common_corpus = [dictionary.doc2bow(text) for text in corpus]
    topics = LdaModel(common_corpus, num_topics=n_topics, id2word=dictionary)
    return topics

# Descomentar para testar as funcionalidades
if __name__ == '__main__':
    docs = get_texts_from_file('../data/febre_amarela_jun17-out18.json',
                                 'title')
    corpus = tokenize_texts(docs)
    processed_corpus = stemmize_text(remove_stopwords(corpus))
    topics = get_topics(processed_corpus,3)  
    for idx, topic in topics.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

