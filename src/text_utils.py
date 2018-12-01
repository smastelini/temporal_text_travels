import pandas as pd
import numpy as np
import re
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_text_and_time_data_from_file(file_path):
    """Read text from 'file_path' and returns the content in 'column_name as
       a list of strings.
    """
    data = pd.read_json(file_path)
    data_ = data.loc[:, ['date', 'title']]

    data_['date'] = pd.to_datetime(data_['date'])
    return data_


def remove_duplicated_texts(data):
    """Remove duplicated text entries."""
    seen = set()
    ut_ids = []
    for i, text in enumerate(data['title'].values):
        if text not in seen:
            seen.add(text)
            ut_ids.append(i)
    return data.iloc[np.array(ut_ids), :]


def tokenize_texts(texts):
    """Remove invalid characters and tokenize texts."""
    texts = [re.sub('[^A-Za-zçáéíóúàããẽõ]', ' ', text).lower()
             for text in texts]
    texts = [word_tokenize(text, language='portuguese') for text in texts]
    return texts


def remove_stopwords(texts):
    """Remove stopwords from the lists of tokens"""
    stopwords_ = stopwords.words('portuguese')
    stopwords_.extend(['é','da','do','de'])
    processed_texts = []
    for text in texts:
        ctext = text.copy()
        for word in text:
            if word in stopwords_:
                ctext.remove(word)
        processed_texts.append(ctext)
    return processed_texts


def stemmize_text(texts):
    """Stemmize each token in the list of tokens"""
    # Stemming and correct spelling
    stemmer = RSLPStemmer()
    texts_ = []
    for text in texts:
        text_ = [stemmer.stem(w) for w in text]
        texts_.append(text_)
    return texts_


def untokenize_text(texts):
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

def split_events(data, init, end, freq='W'):
    """ Splits events given an initial ('init') and ending ('end') time.

    The time windows are defined by 'interval'. The possible values are
    'D': per day, 'W': per week, and 'M': per month.
    """
    slices_raw = [g.reset_index() for n, g in
                  data.set_index('date').groupby(pd.Grouper(freq=freq))]
    slices = [slc for slc in slices_raw if slc.shape[0] > 0]
    return slices

def get_topics_lda(corpus,dictionary,n_topics,passes):
    common_corpus = [dictionary.doc2bow(text) for text in corpus]
    topics = LdaModel(common_corpus, num_topics=n_topics, id2word=dictionary, passes=passes)
    return topics

def get_topics_lsi(corpus,dictionary,n_topics):
    common_corpus = [dictionary.doc2bow(text) for text in corpus]
    topics = LsiModel(common_corpus, num_topics=n_topics, id2word=dictionary)
    return topics

def get_relevant_terms(topics_model,dictionary,n_terms):
    relevant = []
    for idx in range(n_topics):
        tt = topics_model.get_topic_terms(idx,n_terms)
        relevant.extend([dictionary[pair[0]] for pair in tt])
    return set(relevant)

def get_term_document_matrix(corpus,terms):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    tdm = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    return tdm.loc[:,terms]

def get_adj_matrix(corpus,terms):
    tdm = get_term_document_matrix(corpus['title'].values,terms)
    adj_matrix = tdm.transpose().dot(tdm)
    np.fill_diagonal(adj_matrix.values, 0)
    adj_matrix=adj_matrix.fillna(value=0)
    return adj_matrix

def get_splitted_adj_matrix(splitted_events,terms):
    splitted_adj_matrix = []
    for events in splitted_events:
        adj_matrix = get_adj_matrix(events,terms)
        
        splitted_adj_matrix.append(adj_matrix)
    return splitted_adj_matrix

# Descomentar para testar as funcionalidades
if __name__ == '__main__':
     corpus = get_text_and_time_data_from_file('../data/febre_amarela_jun17-out18.json')
     n_passes_lda = 100
     n_topics = 4
     n_terms = 10
     freq_split = 'M'   
     
     #====================================Pre processing========================================   
     corpus = remove_duplicated_texts(corpus)
     tokenized_corpus = tokenize_texts(corpus['title'].values)
     processed_corpus = stemmize_text(remove_stopwords(tokenized_corpus))
     
     #====================================Extract relevant terms================================   
     dictionary = Dictionary(processed_corpus)
     topics = get_topics_lda(processed_corpus,dictionary,n_topics,n_passes_lda)   
     relevant_terms = get_relevant_terms(topics,dictionary,n_terms)
     
     #=======================Generate adjacency matrix per splits===============================
     corpus['title'] = untokenize_text(processed_corpus)
     splitted_corpus = split_events(corpus, corpus['date'].min(), corpus['date'].max(),freq=freq_split)   
     splitted_adj_matrix = get_splitted_adj_matrix(splitted_corpus,relevant_terms)
     
     print(splitted_adj_matrix[10])