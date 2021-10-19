# streamlit import
import streamlit as st
import base64

# backend import
from gensim.models import KeyedVectors
import json
import re
import numpy as np
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymorphy2 import MorphAnalyzer
import torch
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import time

# functions for backend
def tokenize_lemmatize(doc, output_style='line'):
    tokens = tokenizer.tokenize(doc)
    tokens = [morph.parse(token)[0].normal_form for token in tokens if morph.parse(token)[0].normal_form not in stops]
    tokens = [token for token in tokens if re.search('[А-яЁё]', token)]
    preprocessed_line = ' '.join(tokens)
    if output_style == 'list':
        return tokens
    else:
        return preprocessed_line


def fasttext_vec(docs):

    vectorized = []
    for doc in docs:
        token_vecs = []
        for token in doc:
            token_vec = fasttext_model[token]
            token_vecs.append(token_vec)
        if len(token_vecs) == 0:
            doc_vec = np.zeros(300)
        else:
            doc_vec = sum(token_vecs) / len(token_vecs)
        vectorized.append(doc_vec)

    return np.asarray(vectorized)

def mean_pooling(model_output):
    return model_output[0][:, 0]

def bert_vec(docs):
    encoded_input = bert_tokenizer(docs, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    return mean_pooling(model_output)


def preprocess_docs(docs, vectorization='fasttext'):
    if vectorization == 'fasttext':
        tl_docs = [tokenize_lemmatize(doc, output_style='list') for doc in docs]
        matrix = fasttext_vec(tl_docs)

    elif vectorization == 'bert':
        matrix = bert_vec(docs)

    elif vectorization == 'bm25':
        tl_docs = [tokenize_lemmatize(doc) for doc in docs]
        count_vectorizer = CountVectorizer()
        tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

        x_count_vec = count_vectorizer.fit_transform(tl_docs)
        x_tf_vec = tf_vectorizer.fit_transform(tl_docs)
        x_tfidf_vec = tfidf_vectorizer.fit_transform(tl_docs)

        idf = tfidf_vectorizer.idf_
        idf = np.expand_dims(idf, axis=0)
        tf = x_tf_vec

        x_count_vec = count_vectorizer.fit_transform(tl_docs)
        x_tf_vec = tf_vectorizer.fit_transform(tl_docs)
        x_tfidf_vec = tfidf_vectorizer.fit_transform(tl_docs)

        idf = tfidf_vectorizer.idf_
        idf = np.expand_dims(idf, axis=0)
        tf = x_tf_vec

        k = 2
        b = 0.75

        len_d = x_count_vec.sum(axis=1)
        avdl = len_d.mean()

        values = []
        rows = []
        cols = []

        len_d = x_count_vec.sum(axis=1)
        avdl = len_d.mean()
        B_1 = (k * (1 - b + b * len_d / avdl))
        B_1 = np.expand_dims(B_1, axis=-1)

        for i, j in zip(*tf.nonzero()):
            A = tf[i, j] * idf[0][j] * (k + 1)
            B = tf[i, j] + B_1[i]
            value = A / B
            values.append(value[0][0])
            rows.append(i)
            cols.append(j)

        sparse_matrix = sparse.csr_matrix((values, (rows, cols)))

        return sparse_matrix, count_vectorizer

    elif vectorization == 'tfidf':

        tl_docs = [tokenize_lemmatize(doc) for doc in docs]
        tfidf_vectorizer = TfidfVectorizer()
        x_tfidf_vec = tfidf_vectorizer.fit_transform(tl_docs)
        return x_tfidf_vec, tfidf_vectorizer

    else:
        tl_docs = [tokenize_lemmatize(doc) for doc in docs]
        count_vectorizer = CountVectorizer()
        x_count_vec = count_vectorizer.fit_transform(tl_docs)
        return x_count_vec, count_vectorizer

    return matrix

def index_query(query, vectorization='fasttext', vectorizer=None):
    if vectorization == 'fasttext':
        indexed_query = preprocess_docs([query], vectorization)
        indexed_query = np.squeeze(np.asarray(indexed_query))
        return np.expand_dims(indexed_query, axis=0)
    elif vectorization == 'bert':
        indexed_query = preprocess_docs([query], vectorization)
        return indexed_query
    else:
        prepocessed = tokenize_lemmatize(query)
        return vectorizer.transform([query])


def cos_similarity(source_matrix, indexed_query, vectorization='fasttext'):
    if vectorization == 'bert':
        exp = np.expand_dims(indexed_query[0], axis=0)
        cm_matrix = cosine_similarity(source_matrix.numpy(), exp)
    if vectorization != 'bm25':
        cm_matrix = cosine_similarity(source_matrix, indexed_query)
    else:
        cm_matrix = np.dot(source_matrix, indexed_query.T).toarray()

    return cm_matrix

def search_query(preprocessed_corpus, query, docs_array, vectorization='fasttext', vectorizer=None):
    indexed_query = index_query(query, vectorization, vectorizer)
    scores = cos_similarity(preprocessed_corpus, indexed_query, vectorization)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return docs_array[sorted_scores_indx.ravel()][:5], np.sort(scores)[::-1][:5]

def find_answer(query, vectorization, docs_array):

    if vectorization == 'bert':
        return search_query(bert_corpus, query, docs_array, vectorization, None)
    elif vectorization == 'fasttext':
        return search_query(fasttext_corpus, query, docs_array, vectorization, None)
    elif vectorization == 'bm25':
        return search_query(bm25_corpus, query, docs_array, vectorization, count_vectorizer)
    elif vectorization == 'tfidf':
        return search_query(tfidf_corpus, query, docs_array, vectorization, tfidf_vectorizer)
    else:
        return search_query(countvectorizer_corpus, query, docs_array, vectorization, count_vectorizer)

# functions for frontend
def answer(url):
    st.markdown(f'<p style="background-color:#ffffff80;color:#31333F;font-size:16px;border-radius:0.5%; ">{url}</p>',
                    unsafe_allow_html=True)

# cache
@st.cache(allow_output_mutation=True)
def input_models():
    with open('bert_corpus.pickle', 'rb') as f:
        bert_corpus = pickle.load(f)
    with open('bert_model.pickle', 'rb') as f:
        bert_model = pickle.load(f)
    with open('bert_tok.pickle', 'rb') as f:
        bert_tokenizer = pickle.load(f)

    tokenizer = RegexpTokenizer(r'[\w-]+')
    morph = MorphAnalyzer()
    stops = set(stopwords.words('russian'))

    with open('fasttext_corpus.pickle', 'rb') as f:
        fasttext_corpus = pickle.load(f)

    with open('tfidf_corpus.pickle', 'rb') as f:
        tfidf_corpus = pickle.load(f)
    with open('tfidf_vectorizer.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open('countvectorizer_vectorizer.pickle', 'rb') as f:
        count_vectorizer = pickle.load(f)

    with open('bm25_corpus.pickle', 'rb') as f:
         bm25_corpus = pickle.load(f)

    with open('countvectorizer_corpus.pickle', 'rb') as f:
        countvectorizer_corpus = pickle.load(f)

    fasttext_model = KeyedVectors.load('model.model')

    with open('questions.pickle', 'rb') as f:
        docs_array = pickle.load(f)

    return bert_corpus, bert_model, bert_tokenizer, tokenizer, morph, stops, fasttext_corpus, tfidf_corpus,\
           tfidf_vectorizer, count_vectorizer, bm25_corpus, countvectorizer_corpus, fasttext_model, docs_array

# main code
bert_corpus, bert_model, bert_tokenizer, tokenizer, morph, stops, fasttext_corpus, tfidf_corpus, tfidf_vectorizer, count_vectorizer, bm25_corpus, countvectorizer_corpus, fasttext_model, docs_array = input_models()

st.title('Вопросы Дэндзи о любви')
st.header('Настя Панасюк, 182')
st.write("""Дэндзи 16 лет, и он очень многого не знает. Задайте вопрос о любви.""")

main_bg = "sample2.jpg"
main_bg_ext = "jpg"

query = st.text_input('Задать вопрос о любви')
l_col, r_col = st.columns(2)
search_type = l_col.selectbox('Метод', ['countvectorizer', 'tfidf', 'bm25', 'fasttext', 'bert'])

if st.button('Искать ответ'):
    start = time.time()
    output = find_answer(query, search_type, docs_array)
    end = time.time()
    st.write(f'За: {round(end - start, 3)} с')
    for a in output:
        if len(a) > 50:
            answer(a)
        else:
            st.write(a)
