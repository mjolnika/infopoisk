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
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings


def pick_docs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:10000]
    docs = []
    questions = []
    for doc in corpus:
        max_val = 0
        answers = json.loads(doc)['answers']
        for answer in answers:
            answ_val = 0
            try:
                answ_val = int(answer['author_rating']['value'])
            except ValueError:
                pass
            if answ_val > max_val:
                max_val = answ_val
                text = answer['text']
        questions.append(json.loads(doc)['question'])
        docs.append(text)

    return docs, questions

def preprocess_corpus(filename, vectorization='fasttext'):
    
    docs, questions = pick_docs(filename)
    docs_array = np.array(docs)

    if vectorization in ['bm25', 'tfidf', 'countvectorizer']:
        preprocessed_docs, vectorizer = preprocess_docs(docs, vectorization)
        return preprocessed_docs, docs_array, questions, vectorizer
    else:
        preprocessed_docs = preprocess_docs(docs, vectorization)
        return preprocessed_docs, docs_array, questions, None

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
    model = KeyedVectors.load('model.model')

    vectorized = []
    for doc in docs:
        token_vecs = []
        for token in doc:
            token_vec = model[token]
            token_vecs.append(token_vec)
        if len(token_vecs) == 0:
            doc_vec = np.zeros(300)
        else:
            doc_vec = sum(token_vecs)/len(token_vecs)
        vectorized.append(doc_vec)

    return np.matrix(vectorized)

def mean_pooling(model_output):
        return model_output[0][:,0]

def bert_vec(docs):
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    
    encoded_input = tokenizer(docs, padding=True, truncation=True, max_length=24, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

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
            A = tf[i, j]*idf[0][j]*(k + 1)
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
    return docs_array[sorted_scores_indx.ravel()][:10]

def index_matrix(matrix, vectorizer=None):
    tl_matrix = [tokenize_lemmatize(doc) for doc in matrix]
    return vectorizer.transform(tl_matrix)

def count_score(corpus, questions, vectorization):
    if vectorization == 'bert':
        scores = np.array(torch.matmul(corpus, questions.T))
    else:
        scores = cosine_similarity(corpus, questions)

    count = 0
    for i in range(len(scores)):
        sorted_scores = np.argsort(scores[i], axis=0)[::-1]
        if i in sorted_scores[:5]:
            count += 1

    return count/len(scores)


if __name__ == "__main__":

    # task 1
    filename = 'questions_about_love.jsonl'
    vec_types = 'countvectorizer, tfidf, bm25, fasttext, bert'.split(', ')
    vectorization = input('countvectorizer, tfidf, bm25, fasttext, bert?')

    if vectorization not in vec_types:
        vectorization = input('countvectorizer, tfidf, bm25, fasttext, bert?')

    tokenizer = RegexpTokenizer(r'[\w-]+')
    morph = MorphAnalyzer()
    stops = set(stopwords.words('russian'))

    preprocessed_corpus, docs_array, questions, vectorizer = preprocess_corpus(filename, vectorization)
    
    query = input('Введите запрос: ')
    while query != 'None':
        for d in search_query(preprocessed_corpus, query, docs_array, vectorization, vectorizer):
          print(d)
        query = input('Введите запрос: ')
        
    # task 2
    proceed = input('proceed to task 2? [y/n]')
    if proceed == 'y':
        for vec_type in vec_types:
            vectorization = vec_type
            if vectorization in ['countvectorizer', 'tfidf', 'bm25']:
                preprocessed_corpus, docs_array, questions, vectorizer = preprocess_corpus(filename, vectorization)
                preprocessed_questions = index_matrix(questions, vectorizer)
            else:
                preprocessed_corpus, docs_array, questions, vectorizer = preprocess_corpus(filename, vectorization)
                preprocessed_questions = preprocess_docs(questions, vectorization)
            
            scores = count_score(preprocessed_corpus, preprocessed_questions, vectorization)
            #countvectorizer: 0.0526
            #tfidf: 0.0808
            #bm25: 0.0766
            #fasttext: 0.0559
            #bert: 0.0064
            print(f'{vec_type}: {scores}')
