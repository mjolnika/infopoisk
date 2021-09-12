# Anastasia Panasyuk, 182

import os
import re
import nltk
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm.auto import tqdm
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

tokenizer = RegexpTokenizer(r'[\w-]+')
morph = MorphAnalyzer()
vectorizer = CountVectorizer(analyzer='word')
stops = set(stopwords.words('russian'))


heroes = '''Моника, Мон
Рэйчел, Рейч
Чендлер, Чэндлер, Чен
Фиби, Фибс
Росс
Джоуи, Джои, Джо'''.lower().replace(' ', '').split('\n')
heroes = [hero.split(',') for hero in heroes]


filenames = []
for path, dirs, files in os.walk('friends-data'):
    for file in files:
        filenames.append(os.path.join(path, file))
        
        
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
invX = X.T

max_freq = 0
max_id = 0
min_freq = 100
min_id = 0
every_doc = []
for n, term in enumerate(invX):
    freq = term.toarray()[0].sum()
    if freq > max_freq:
        max_freq = freq
        max_id = n
    elif freq < min_freq:
        min_freq = freq
        min_id = n
    if 0 not in term.toarray()[0]:
        every_doc.append(n)
every_doc_display = [feature_names[n_id] for n_id in every_doc]


print('Maximum frequency:', max_freq, feature_names[max_id])
print('Minimum frequency:', min_freq, feature_names[min_id])
print('In every document:', ', '.join(every_doc_display))
print('Если отсекать стоп-слова (весь, твой и так далее), самое популярное - Росс, 1016, что также ответ на след. вопрос')

hero_dict = {}
for hero in heroes:
    hero_dict[hero[0]] = [vectorizer.vocabulary_.get(variant) for variant in hero
                          if vectorizer.vocabulary_.get(variant) is not None]
    
hero_freq = {}
for key, values in hero_dict.items():
    mentions = [invX[value][0].toarray().sum() for value in values]
    freq = sum(mentions)
    hero_freq[key] = freq

for k, v in sorted(hero_freq.items(), key = lambda item: item[1], reverse=True):
    print(k, v)
    
print('Чаще всего упоминается Росс')
