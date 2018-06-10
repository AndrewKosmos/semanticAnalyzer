import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle
import nltk as nltk
import pickle

import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence

from keras.callbacks import EarlyStopping
from keras.models import Sequential
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

tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = SnowballStemmer("russian")

def ingest():
    data = pd.read_csv('./data_set.csv')
    data['label'] = data['label'].map(int)
    data = data[data['text'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('id', axis=1, inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape)   
    return data

data = ingest()
#print(data)
def tokenize(tweet):
    try:
        tweet = tweet.strip().lower()
        tokens_t = nltk.word_tokenize(tweet)
        #tokens_t = [j for j in tokens_t if ( j not in string.punctuation )]
        stop_words = stopwords.words('russian')
        stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
        tokens_t = [j for j in tokens_t if ( j not in stop_words )]
        tokens_t = [j.replace("«", "").replace("»", "") for j in tokens_t]
        tokens = [stemmer.stem(t) for t in tokens_t if not t.startswith('@')]
        return tokens
    except:
        return 'NC'

def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['text'].map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)
#print(data)

n=1000000
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).label), test_size=0.2)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

tweet_w2v = Word2Vec(size=200, min_count=10)
tweet_w2v.build_vocab([x.words for x in x_train])
tweet_w2v.train([x.words for x in x_train],total_examples=tweet_w2v.corpus_count, epochs=100)

tweet_w2v.save("w2vmodel.model")

print ('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))
pickle.dump(tfidf, open("tfidf.pickle", "wb"))


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

n_dim = 200
from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, x_train)])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, x_test)])
test_vecs_w2v = scale(test_vecs_w2v)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=10, batch_size=32, verbose=2)

score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print (score[1])

model.save('model.h5')