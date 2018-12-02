import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from gensim.models import LdaModel, LsiModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Projection tools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap

# Command line commands manipulation
import sys
import getopt

# Visualization tools
# import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go


def get_text_and_time_data_from_file(file_path):
    """Read text from 'file_path' and returns the content in 'column_name as
       a list of strings.
    """
    data = pd.read_csv(file_path)
    data_ = data.loc[:, ['date', 'title']]

    data_['date'] = pd.to_datetime(data_['date'])
    data_.sort_values(by=['date'])
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


def tokenize_texts(texts, language='portuguese'):
    """Remove invalid characters and tokenize texts."""
    texts = [re.sub('[^A-Za-zçáéíóúàããẽõ]', ' ', text).lower()
             for text in texts]
    texts = [word_tokenize(text, language=language) for text in texts]
    return texts


def remove_stopwords(texts, domain_stopwords=None, language='portuguese'):
    """Remove stopwords from the lists of tokens"""
    stopwords_ = stopwords.words(language)
    if language == 'portuguese':
        stopwords_.extend(['é', 'da', 'do', 'de'])
    if domain_stopwords is not None:
        stopwords_.extend(domain_stopwords)
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
    # To count the most common for of a root term
    root2frequent = {}
    # Stemming and correct spelling
    stemmer = RSLPStemmer()
    texts_ = []
    for text in texts:
        text_ = []
        for w in text:
            stem = stemmer.stem(w)
            try:
                root2frequent[stem].update({w: 1})
            except KeyError:
                root2frequent[stem] = Counter()
                root2frequent[stem].update({w: 1})
            text_.append(stem)
        texts_.append(text_)
    return texts_, root2frequent


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
    # slices = [slc for slc in slices_raw if slc.shape[0] > 0]
    return slices_raw


def get_topics_lda(corpus, dictionary, n_topics, passes):
    common_corpus = [dictionary.doc2bow(text) for text in corpus]
    topics = LdaModel(common_corpus, num_topics=n_topics, id2word=dictionary,
                      passes=passes)
    return topics


def get_topics_lsi(corpus, dictionary, n_topics):
    common_corpus = [dictionary.doc2bow(text) for text in corpus]
    topics = LsiModel(common_corpus, num_topics=n_topics, id2word=dictionary)
    return topics


def get_relevant_terms(topics_model, dictionary, n_topics, n_terms):
    relevant = []
    for idx in range(n_topics):
        tt = topics_model.get_topic_terms(idx, n_terms)
        relevant.extend([dictionary[pair[0]] for pair in tt])
    return list(set(relevant))


def get_term_document_matrix(corpus, terms):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    tdm = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    return tdm.reindex(labels=terms, axis=1)


def get_adj_matrix(corpus, terms):
    tdm = get_term_document_matrix(corpus['title'].values, terms)
    adj_matrix = tdm.transpose().dot(tdm)
    np.fill_diagonal(adj_matrix.values, 0)
    adj_matrix = adj_matrix.fillna(value=0)
    return adj_matrix


def get_splitted_adj_matrix(splitted_events, terms):
    splitted_adj_matrix = []
    for events in splitted_events:
        if events.shape[0] > 0:
            adj_matrix = get_adj_matrix(events, terms)
        else:
            adj_matrix = np.zeros(0)
        splitted_adj_matrix.append(adj_matrix)
    return splitted_adj_matrix


def join_time_variant_adj_matrices(splitted_adj_matrix):
    n_rows = len(splitted_adj_matrix)
    n_cols = splitted_adj_matrix[0].shape[0] * splitted_adj_matrix[0].shape[1]
    vec_repr = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        if splitted_adj_matrix[i].shape[0] > 0:
            vec_repr[i, :] = splitted_adj_matrix[i].values.reshape((1, n_cols))

    return vec_repr


def get_top_k_terms_and_time_period(splitted_adj_matrix, slices, terms,
                                    unstemmizer, k=5):
    top_terms = []
    for m, s in zip(splitted_adj_matrix, slices):
        if s.shape[0] > 0:
            top_k = m.sum(axis=1).argsort()[-k:].tolist()
            top_ = [unstemmizer[terms[t]].most_common(1)[0][0] for t in top_k]
            if s.shape[0] > 1:
                period = '{0}/{1}/{2}-{3}/{4}/{5}'.format(
                    s.loc[0, 'date'].day,
                    s.loc[0, 'date'].month,
                    s.loc[0, 'date'].year,
                    s.loc[s.shape[0] - 1, 'date'].day,
                    s.loc[s.shape[0] - 1, 'date'].month,
                    s.loc[s.shape[0] - 1, 'date'].year,
                )
            else:
                period = '{0}/{1}/{2}'.format(
                    s.loc[0, 'date'].day,
                    s.loc[0, 'date'].month,
                    s.loc[0, 'date'].year,
                )
            top_terms.append((period, top_))
        else:
            top_terms.append((None, []))
    return top_terms


def project_data_points(data, method='PCA'):
    """ Projects data matrix using a Multidimensional Projection algorithm.

    The points in 'data' are mapped to bidimensional representations, which are
    returned by the function. The supported methods are 'PCA', 't-SNE', 'MDS'
    and 'Isomap'.
    """
    projected = None

    if method == 't-SNE':
        projection = TSNE(n_components=2)
    elif method == 'MDS':
        projection = MDS(n_components=2)
    elif method == 'Isomap':
        projection = Isomap(n_components=2)
    else:
        data = StandardScaler().fit_transform(data)
        projection = PCA(n_components=2)

    projected = projection.fit_transform(data)
    return projected


def plot_projections(projections, top_terms, dataset_name, method,
                     out_path=None):
    """ Creates an interative scatter plot of the projections.
    """
    # Joins the top terms per time slice with line breaks
    formatted_terms = ['<br>'.join(t[1]) if t[0] is not None else ''
                       for t in top_terms]

    trace = go.Scatter(
        x=projections[:, 0],
        y=projections[:, 1],
        mode='lines+markers',
        marker=dict(
            size=15,
            line=dict(
                width=0.3,
                color='rgb(0, 0, 0)'
            ),
            cmin=0,
            cmax=projections.shape[0],
            color=[c for c in range(projections.shape[0])],
            colorbar=dict(
                title='Time'
            ),
            colorscale='Viridis'
        ),
        hoverinfo='text',
        text=['Step {}<br>({}):<br>{}'.format(t, top_terms[t][0],
              formatted_terms[t]) if top_terms[t][0] is not None else
              'Step {}'.format(t) for t in range(projections.shape[0])],
        textposition='top left'
    )

    layout = go.Layout(
        title='{} - {}'.format(dataset_name, method),
        xaxis=dict(
            title='Component 1'
        ),
        yaxis=dict(
            title='Component 2'
            # ticklen=5,
            # gridwidth=2,
        )
    )

    plot_data = {
        'data': [trace],
        'layout': layout
    }

    plot(
        plot_data,
        filename=out_path + '/main.html',
        auto_open=False
    )


# Descomentar para testar as funcionalidades
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:m:s:o:t:e:p:d:',
                                   ['help', 'input=', 'method=', 'slice=',
                                    'output=', 'n_topics=', 'n_terms=',
                                    'n_passes=', 'domain_stopwords='])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    if len(opts) < 1:
        print(
            '\nAt least the input dataset and the output path must be ' +
            'defined. See --help or -h for help.\n\n'
        )
        exit()

    input = None
    output = None
    method = 'PCA'
    time_slice = 'W'
    n_topics = 4
    n_terms = 10
    n_passes_lda = 100
    domain_stopwords = None

    for o, a in opts:
        if o in ('-h', '--help'):
            print(
                '\nVisualizing tendencies in websensors\' events (Beta)\n\n' +
                'Tool for visualizing evolution of term-tendency ' +
                'through time.\n\nParameters:\n\n' +
                '(-i, --input=) Path for the input dataset (csv).\n' +
                '(-s, --slice=) Time slicing strategy:\n' +
                '\tD - Day\n\tW - Week (Default)\n\tM - Month\n' +
                '(-m, --method=) Multidimensional projection method: ' +
                '{PCA, t-SNE, MDS, Isomap} (Default: PCA)\n' +
                '(-t, --n_topics=) Number of topics for the LDA algorithm ' +
                '(Default: 4)\n' +
                '(-e, --n_terms=) Number of terms for the LDA algorithm ' +
                '(Default: 10)\n' +
                '(-p, --n_passes=) Number of passes for the LDA algorithm ' +
                '(Default: 100)\n' +
                '(-d, --domain_stopwords=) List of additional domain only ' +
                'stopwords separated by colons (Example: some:simple:' +
                'example).\n'
                '(-o, --output=) Path for the output folder.\n'
            )
            exit()
        elif o in ('-i', '--input'):
            input = a
        elif o in ('-m', '--method'):
            method = a
        elif o in ('-s', '--slice'):
            time_slice = a
        elif o in ('-t', '--n_topics'):
            n_topics = int(a)
        elif o in ('-e', '--n_terms'):
            n_terms = int(a)
        elif o in ('-p', '--n_passes'):
            n_passes_lda = int(a)
        elif o in ('-d', '--domain_stopwords'):
            domain_stopwords = a.split(':')
        elif o in ('-o', '--output'):
            output = a
        else:
            assert False, 'Unhandled option: {}.'.format(o)
    print('Reading data.')
    corpus = get_text_and_time_data_from_file(input)

    # ==============================Pre processing=============================
    print('Removing duplicated events.')
    corpus = remove_duplicated_texts(corpus)
    print('Tokenizing terms, removing stopwords and applying stemming.')
    tokenized_corpus = tokenize_texts(corpus['title'].values)
    processed_corpus, unstemmizer = stemmize_text(
        remove_stopwords(tokenized_corpus, domain_stopwords)
    )
    # =========================Extract relevant terms==========================
    print('Getting the most relevant terms.')
    dictionary = Dictionary(processed_corpus)
    topics = get_topics_lda(processed_corpus, dictionary, n_topics,
                            n_passes_lda)
    relevant_terms = get_relevant_terms(topics, dictionary, n_topics, n_terms)

    # ================Generate adjacency matrix per splits=====================
    print('Generating dynamic network representation.')
    corpus['title'] = untokenize_text(processed_corpus)
    splitted_corpus = split_events(corpus, corpus['date'].min(),
                                   corpus['date'].max(),
                                   freq=time_slice)
    splitted_adj_matrix = get_splitted_adj_matrix(splitted_corpus,
                                                  relevant_terms)
    print('Getting dynamic network\'s vectorial representation.')
    # Vectorial to be projected in 2D
    vec_repr = join_time_variant_adj_matrices(splitted_adj_matrix)
    # Data projection
    print('Projecting data points.')
    plot_data = project_data_points(vec_repr, method)
    # Get top k terms by window time (by their node degree) and the respective
    # time period
    print('Ploting data.')
    top_terms = get_top_k_terms_and_time_period(
        splitted_adj_matrix,
        splitted_corpus,
        relevant_terms,
        unstemmizer,
        k=5
    )

    dataset_name = input.split('/')[-1]
    if not os.path.exists(output):
        os.makedirs(output)
    plot_projections(plot_data, top_terms, dataset_name, method, output)
    print('Operations finished.')
