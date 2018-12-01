import pandas as pd
import numpy as np
import re
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Projection tools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap

# Command line commands manipulation
import sys
import getopt

# Visualization tools
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go


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


def tokenize_texts(texts, language='portuguese'):
    """Remove invalid characters and tokenize texts."""
    texts = [re.sub('[^A-Za-zçáéíóúàããẽõ]', ' ', text).lower()
             for text in texts]
    texts = [word_tokenize(text, language=language) for text in texts]
    return texts


def remove_stopwords(texts, language='portuguese'):
    """Remove stopwords from the lists of tokens"""
    stopwords_ = stopwords.words(language)
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


def get_topics(corpus, n_topics):
    dictionary = Dictionary(corpus)
    common_corpus = [dictionary.doc2bow(text) for text in corpus]
    topics = LdaModel(common_corpus, num_topics=n_topics, id2word=dictionary)
    return topics


def split_events(data, init, end, freq='W'):
    """ Splits events given an initial ('init') and ending ('end') time.

    The time windows are defined by 'interval'. The possible values are
    'D': per day, 'W': per week, and 'M': per month.
    """
    slices_raw = [g.reset_index() for n, g in
                  data.set_index('date').groupby(pd.Grouper(freq=freq))]
    slices = [slc for slc in slices_raw if slc.shape[0] > 0]
    return slices


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
    # Joins the top terms per time slice with commas and space
    formatted_terms = [', '.join(t) for t in top_terms]

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
        text=['{}: {}'.format(t, formatted_terms[t])
              for t in range(projections.shape[0])],
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
        opts, args = getopt.getopt(sys.argv[1:], 'hi:m:s:o:',
                                   ['help', 'input=', 'method=', 'slice=',
                                    'output='])
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
    for o, a in opts:
        if o in ('-h', '--help'):
            print(
                '\nVisualizing tendencies in websensors\' events (Beta)\n\n' +
                'Tool for visualizing inter-target correlations ' +
                'through time.\n\nParameters:\n\n' +
                '(-i, --input=) Path for the input dataset (csv).\n' +
                '(-s, --slice=) Time slicing strategy:\n' +
                '\tD - Day\n\tW - Week (Default)\n\tM - Month\n' +
                '(-m, --method=) Multidimensional projection method: ' +
                '{PCA, t-SNE, MDS, Isomap}\n' +
                '(-o, --output=) Path for the output folder.\n'
            )
            exit()
        elif o in ('-i', '--input'):
            input = a
        elif o in ('-m', '--method'):
            method = a
        elif o in ('-s', '--slice'):
            time_slice = a
        elif o in ('-o', '--output'):
            output = a
        else:
            assert False, 'Unhandled option: {}.'.format(o)

    # docs = get_text_and_time_data_from_file(
    #     '../data/febre_amarela_jun17-out18.json'
    # )
    # 
    # docs = remove_duplicated_texts(docs)
    # 
    # corpus = tokenize_texts(docs['title'].values)
    # processed_corpus = stemmize_text(remove_stopwords(corpus))
    # split_events(docs, docs['date'].min(), docs['date'].max())
    # topics = get_topics(processed_corpus, 3)
    # for idx, topic in topics.print_topics(-1):
    #     print('Topic: {} \nWords: {}'.format(idx, topic))
