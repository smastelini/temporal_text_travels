import os

python_command = 'python3'
datasets = {
    'dengue_pt': 'dengue:globo',
    'febre_amarela': 'febre:amarela:globo',
    'trump_pt': 'donald:trump:the:globo'
}

time_divisions = ['W', 'M']

projection_techniques = ['PCA', 't-SNE', 'PCA-T']
output_prefix = '../results'

number_of_topics = 4
number_of_terms = 10
number_of_passes = 100

for tech in projection_techniques:
    for data_name, domain_stopwords in datasets.items():
        for slice in time_divisions:
            command_string = '{0} temporal_term_visualization.py -i '\
                '../data/{1}.csv -o {2}/{1}_{3}_ntopics'\
                '_{4}_nterms_{5}_npasses_{6}_per_{8} -m {3} -t {4} -e {5} '\
                '-p {6} -d {7} -s {8}'.format(
                    python_command, data_name, output_prefix, tech,
                    number_of_topics, number_of_terms, number_of_passes,
                    domain_stopwords, slice
                )
            os.system(command_string)
