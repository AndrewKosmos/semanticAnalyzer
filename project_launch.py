import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from copy import deepcopy
import string
from string import punctuation
from random import shuffle
import nltk as nltk
import pickle
import sys

import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D

from tqdm import tqdm
#tqdm.pandas(desc="progress-bar")
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.nist import NISTTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

w2v = Word2Vec.load("w2vmodel.model")
model = load_model('model.h5')
tfidf = pickle.load(open("tfidf.pickle", "rb"))
n_dim = 200

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

stemmer = SnowballStemmer("russian")
cv = nltk.word_tokenize(sys.argv[1:][0])
cv = [j for j in cv if ( j not in string.punctuation )]
stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
cv = [j for j in cv if ( j not in stop_words )]
cv = [j.replace("«", "").replace("»", "") for j in cv]
print(cv)


vecs = []
tokens = [stemmer.stem(t) for t in cv if not t.startswith('@')] 
from sklearn.preprocessing import scale
predict_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tokens])
predict_vecs_w2v = scale(predict_vecs_w2v)

#print(predict_vecs_w2v)
score = model.predict_classes(predict_vecs_w2v,verbose=1)
print(score.mean().astype(np.float32))
#print(model.predict_classes(predict_vecs_w2v,verbose=1))
#score = model.predict(predict_vecs_w2v)
#print(score.mean().astype(np.float32))
#print(model.predict(predict_vecs_w2v))