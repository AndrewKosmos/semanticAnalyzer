import np
import sys
import string
import nltk as nltk
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from nltk.stem.snowball import SnowballStemmer
from keras.models import load_model
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.nist import NISTTokenizer
from nltk.corpus import stopwords


dataset_location = 'data_set.csv'
model_location = 'model/'

corpus = []
labels = []

with open(dataset_location, 'r', encoding='utf-8') as df:
    for i, line in enumerate(df):
        if i == 0:
            continue

        parts = line.strip().split(',')
        
        # Sentiment (0 = Negative, 1 = Positive)
        labels.append(int(parts[2].strip()))
        
        # Tweet
        tweet = parts[1].strip()
        if tweet.startswith('"'):
            tweet = tweet[1:]
        if tweet.endswith('"'):
            tweet = tweet[::-1]
        
        corpus.append(tweet.strip().lower())

print('Corpus size: {}'.format(len(corpus)))

tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = SnowballStemmer("russian")

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokenized_twt = nltk.word_tokenize(tweet)
    tokenized_twt = [j for j in tokenized_twt if ( j not in string.punctuation )]
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
    tokenized_twt = [j for j in tokenized_twt if ( j not in stop_words )]
    tokenized_twt = [j.replace("«", "").replace("»", "") for j in tokenized_twt]
    tokens = [stemmer.stem(t) for t in tokenized_twt if not t.startswith('@')]
    #print(tokens)
    tokenized_corpus.append(tokens)

vector_size = 512
window_size = 10

word2vec = Word2Vec(sentences=tokenized_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=50,
                    seed=1000,
                    workers=4)

model = load_model('model.h5')
stemmer = SnowballStemmer("russian")
cv = nltk.word_tokenize(sys.argv[1:][0])
cv = [j for j in cv if ( j not in string.punctuation )]
stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
cv = [j for j in cv if ( j not in stop_words )]
cv = [j.replace("«", "").replace("»", "") for j in cv]


vecs = []
tokens = [stemmer.stem(t) for t in cv if not t.startswith('@')] 
vec = np.zeros(512).reshape((1,512))
count = 0.
for word in tokens:
	try:
		#vec += word2vec[word].reshape((1,512))
		vecs.append(word2vec[word].reshape((1,512)))
		count += 1.
	except KeyError: 
		continue

if count != 0:
	vec /= count

print(model.predict(vecs))

#def buildWordVector(tokens, size):
#    vec = np.zeros(size).reshape((1, size))
#    count = 0.
#    for word in tokens:
#        try:
#            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
#            count += 1.
#        except KeyError: # handling the case where the token is not
#                         # in the corpus. useful for testing.
#            continue
#    if count != 0:
#        vec /= count
#    return vec