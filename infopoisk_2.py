import os
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm.auto import tqdm
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_corpus(directory, vectorizer):
    filenames = []
    names = []
    for path, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(path, file))
            names.append(file)
                 
    docs = []
    for file in tqdm(filenames):
        with open(file, encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.tokenize(text)
        tokens = [morph.parse(token)[0].normal_form for token in tokens if morph.parse(token)[0].normal_form not in stops]
        tokens = [token for token in tokens if re.search('[А-яЁё]', token)]
        docs.append(' '.join(tokens))
        
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()
    return X, feature_names, names, vectorizer

def create_query_matrix(query):
    tokens = tokenizer.tokenize(query)
    tokens = [morph.parse(token)[0].normal_form for token in tokens if morph.parse(token)[0].normal_form not in stops]
    tokens = [token for token in tokens if re.search('[А-яЁё]', token)]
    query = [' '.join(tokens)]
    return vectorizer.transform(query)

def normalize(x):
    return x / np.linalg.norm(x)

def cos_sim_matrix(normalized_matrix, query_matrix):
    cos_sim_m = [cosine_similarity(normalize(query_matrix.toarray()), doc) for doc in normalized_matrix]
    return cos_sim_m

def range_docs(sim_matrix, filenames):
    results = {}
    for i in range(len(filenames)):
        results[filenames[i]] = sim_matrix[i]
    return sorted(results, key = lambda key: results[key], reverse=True)

def find_docs(query, corpus_matrix, vectorizer, filenames):
    query_matrix = create_query_matrix(query)
    normalized_matrix = [normalize(x.toarray()) for x in corpus_matrix]
    cos_sim_m = cos_sim_matrix(normalized_matrix, query_matrix)
    return range_docs(cos_sim_m, filenames)


tokenizer = RegexpTokenizer(r'[\w-]+')
morph = MorphAnalyzer()
vectorizer = TfidfVectorizer(analyzer='word')
stops = set(stopwords.words('russian'))

directory = 'friends-data'
matrix, feature_names, filenames, vectorizer = create_corpus(directory, vectorizer)

query = 'секс, наркотики, рок-н-ролл'
answers = find_docs(query, matrix, vectorizer, filenames)
#print(answers)
