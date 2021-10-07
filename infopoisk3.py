import json
import re
import numpy as np
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def preprocess_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    docs = []
    for doc in corpus:
        max_val = 0
        answers = json.loads(doc)['answers']
        for answer in answers:
            answ_val = 0
            try:
                answ_val = int(answer['author_rating']['value'])
            except ValueError:
                pass
            if  answ_val > max_val:
                max_val = answ_val
                text = answer['text']
        docs.append(text)
        
    preprocessed_docs = []
    for doc in docs:
        preprocessed_docs.append(preprocess_doc(doc))
        
    docs_array = np.array(docs)
    
    return preprocessed_docs, docs_array
    
def index_corpus(filename):
    preprocessed_docs, docs_array = preprocess_corpus(filename)
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    
    x_count_vec = count_vectorizer.fit_transform(preprocessed_docs) 
    x_tf_vec = tf_vectorizer.fit_transform(preprocessed_docs) 
    x_tfidf_vec = tfidf_vectorizer.fit_transform(preprocessed_docs) 
    
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vec
    
    x_count_vec = count_vectorizer.fit_transform(preprocessed_docs)
    x_tf_vec = tf_vectorizer.fit_transform(preprocessed_docs)
    x_tfidf_vec = tfidf_vectorizer.fit_transform(preprocessed_docs)
    
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
    
    return count_vectorizer, sparse_matrix, docs_array
    

def preprocess_doc(doc):
    tokens = tokenizer.tokenize(doc)
    tokens = [morph.parse(token)[0].normal_form for token in tokens if morph.parse(token)[0].normal_form not in stops]
    tokens = [token for token in tokens if re.search('[А-яЁё]', token)]
    preprocessed_line = ' '.join(tokens)
    return preprocessed_line   

def index_query(query):
    prepocessed = preprocess_doc(query)
    return count_vectorizer.transform([query])

def cos_sim(sp_matrix, query_vec):
    return np.dot(sp_matrix, query_vec.T).toarray()

def search_query(query, sp_matrix, docs_array):
    query_vec = index_query(query)
    scores = cos_sim(sp_matrix, query_vec)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return docs_array[sorted_scores_indx.ravel()]
    

if __name__ == "__main__":
    tokenizer = RegexpTokenizer(r'[\w-]+')
    morph = MorphAnalyzer()
    vectorizer = TfidfVectorizer(analyzer='word')
    stops = set(stopwords.words('russian'))
    
    count_vectorizer, sparse_matrix, docs_array = index_corpus('questions_about_love.jsonl')
    query = input('Введите запрос: ')
    while query != 'None':
        print(search_query(query, sparse_matrix, docs_array))
        query = input('Введите запрос: ')
