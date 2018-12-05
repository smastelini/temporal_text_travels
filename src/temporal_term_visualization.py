import pandas as pd
import numpy as np
import re
import os
import math
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
    data = pd.read_csv(file_path, dtype=object)
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
    texts = [re.sub('[^A-Za-zçáéíóúàããẽõôê]', ' ', text).lower()
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
    slices = [slc for slc in slices_raw if slc.shape[0] > 0]
    return slices


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
    selected_slices = []
    for i, events in enumerate(splitted_events):
        adj_matrix = get_adj_matrix(events, terms)
        degree = np.sum(adj_matrix.values)
        if degree > 0.0:
            selected_slices.append(i)
            splitted_adj_matrix.append(adj_matrix)
    return splitted_adj_matrix, selected_slices


def join_time_variant_adj_matrices(splitted_adj_matrix):
    n_rows = len(splitted_adj_matrix)
    n_cols = splitted_adj_matrix[0].shape[0] * splitted_adj_matrix[0].shape[1]
    vec_repr = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        vec_repr[i, :] = splitted_adj_matrix[i].values.reshape((1, n_cols))

    return vec_repr


def get_top_k_terms_and_time_period(splitted_adj_matrix, slices, terms,
                                    unstemmizer, k=5):
    top_terms = []
    for m, s in zip(splitted_adj_matrix, slices):
        mean_ = np.mean(m, axis=1)
        std_ = np.std(m, axis=1)
        top_k = mean_.argsort()[-k:].tolist()
        top_k.reverse()
        top_ = [unstemmizer[terms[t]].most_common(1)[0][0] for t in
                top_k if mean_[t] > 0]
        degrees = [(mean_[t], std_[t]) for t in top_k if mean_[t] > 0]
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
        top_terms.append((period, top_, degrees))
    return top_terms


def project_data_points(data, method='PCA'):
    """ Projects data matrix using a Multidimensional Projection algorithm.

    The points in 'data' are mapped to bidimensional representations, which are
    returned by the function. The supported methods are 'PCA', 't-SNE', 'MDS'
    and 'Isomap'.
    """
    projected = None
    data = StandardScaler().fit_transform(data)
    if method == 't-SNE':
        projection = TSNE(n_components=2)
    elif method == 'MDS':
        projection = MDS(n_components=2)
    elif method == 'Isomap':
        projection = Isomap(n_components=2)
    elif method == 'PCA-T':
        projection = PCA(n_components=1)
    else:
        projection = PCA(n_components=2)

    projected = projection.fit_transform(data)
    return projected


def plot_projections(projections, top_terms, points_size, method,
                     n_rel, out_path=None):
    """ Creates an interative scatter plot of the projections.
    """
    # Joins the top terms per time slice with line breaks
    terms_degree = []
    for t in top_terms:
        pretty_print = []
        words_ = t[1]
        stats = t[2]
        for w, (mu, sigma) in zip(words_, stats):
            pretty_print.append('[{:0.2f} +/- {:0.2f}] {}'.format(mu, sigma,
                                                                  w))
        terms_degree.append(pretty_print)
    formatted_terms = ['<br>'.join(t) for t in terms_degree]

    if projections.shape[1] > 1:
        trace = go.Scatter(
            x=projections[:, 0],
            y=projections[:, 1],
            mode='markers',
            marker=dict(
                opacity=1,
                size=points_size,
                cmin=0,
                cmax=projections.shape[0],
                color=[c for c in range(projections.shape[0])],
                colorbar=dict(
                    title='Time step'
                ),
                colorscale='Viridis'
            ),
            hoverinfo='text',
            text=['Step {}<br>({}):<br>{}'.format(
                  t, top_terms[t][0],
                  formatted_terms[t]) for t in range(projections.shape[0])],
            textposition='top left',
            showlegend=False
        )

        lines = go.Scatter(
            x=projections[:, 0],
            y=projections[:, 1],
            mode='lines',
            hoverinfo='none',
            line=dict(
                width=0.6,
                color=('rgba(180, 180, 180, 1)')
            ),
            showlegend=False
        )

        layout = go.Layout(
            xaxis=dict(
                title='Component 1'
            ),
            yaxis=dict(
                title='Component 2'
            ),
            annotations=[
                dict(
                    x=0.95,
                    y=1.1,
                    xref='paper',
                    yref='paper',
                    xanchor='center',
                    yanchor='top',
                    text='Number of relevant terms: {}'.format(n_rel),
                    showarrow=False,
                )
            ]
        )
    else:
        trace = go.Scatter(
            x=[t for t in range(projections.shape[0])],
            y=projections[:, 0],
            mode='markers',
            marker=dict(
                opacity=1,
                size=12,
                cmin=0,
                cmax=np.max(projections[:, 0]),
                color=projections[:, 0],
                colorscale='Bluered',
                line=dict(
                    width=0.7,
                    color=('rgb(0, 0, 0)')
                )
            ),
            hoverinfo='text',
            text=['Step {}<br>({}):<br>{}'.format(
                  t, top_terms[t][0],
                  formatted_terms[t]) for t in range(projections.shape[0])],
            textposition='top left',
            showlegend=False
        )

        lines = go.Scatter(
            x=[t for t in range(projections.shape[0])],
            y=projections[:, 0],
            mode='lines',
            hoverinfo='none',
            line=dict(
                width=0.7,
                color=('rgba(180, 180, 180, 1)')
            ),
            showlegend=False
        )

        layout = go.Layout(
            xaxis=dict(
                title='Time'
            ),
            yaxis=dict(
                title='Principal Component 1'
            ),
            annotations=[
                dict(
                    x=0.95,
                    y=1.1,
                    xref='paper',
                    yref='paper',
                    xanchor='center',
                    yanchor='top',
                    text='Number of relevant terms: {}'.format(n_rel),
                    showarrow=False,
                )
            ]
        )

    plot_data = {
        'data': [lines, trace],
        'layout': layout
    }

    plot(
        plot_data,
        filename=out_path + '.html',
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
                '{PCA, t-SNE, MDS, Isomap, PCA-T (Time x PC1)} ' +
                '(Default: PCA)\n' +
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
    splitted_adj_matrix, sel_slices = get_splitted_adj_matrix(splitted_corpus,
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
        # Removing slices that do not contain the relevant terms
        [splitted_corpus[t] for t in sel_slices],
        relevant_terms,
        unstemmizer,
        k=len(relevant_terms)
    )

    min_size = 10
    max_size = 40
    sum_degree = vec_repr.sum(axis=1)
    min_degree = np.min(sum_degree)
    max_degree = np.max(sum_degree)
    points_size = [
        math.ceil(
            (s - min_degree)/(max_degree - min_degree) *
            (max_size - min_size) + min_size
        ) for s in sum_degree
    ]

    out_folder = '/'.join(output.split('/')[:-1])
    if len(out_folder) > 0 and not os.path.exists(out_folder):
        os.makedirs(out_folder)
    plot_projections(plot_data, top_terms, points_size,
                     method, len(relevant_terms), output)
    print('Operations finished.')
