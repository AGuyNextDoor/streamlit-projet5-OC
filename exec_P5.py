import streamlit as st
import pandas as pd
import pipreqs
from pandas.plotting import scatter_matrix, andrews_curves

from datetime import datetime

from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict

import numpy as np

from sklearn.externals import joblib
import gensim

import re

import math
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def CleanedInputLemma(x):
    sw = set()
    sw.update(tuple(nltk.corpus.stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(x)
    tags2 = nltk.pos_tag(tokens)
    tokens = [word.lower() for word,pos in tags2 if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    #corpora = [lemmatizer.lemmatize(w.lower()) for w in tokens if not w in list(sw)]
    #corpora = list(set(corpora))
    corpora = list(dict.fromkeys(tokens))

    #corpora = [w for w in tokens if not w in list(sw)]
    #corpora = [s + " " for s in corpora]
    #print(''.join(corpora[:]))

    return corpora

def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list ])
st.title('PROJET 5 - Martin Vielvoye')
print("new print \n \n")
genre = st.sidebar.radio(
    "Select which model to test",
    ('All', 'Random Forest', 'k-Neighbors', 'MLPeceptron'))

'Predicting for *%s* model(s)' % genre

label = "text input"
input = st.text_input(label, value='hello world', key="Test_input_key")
print("input is : ", input)
if(input == ""): input = "hello world"
if(input != ""):
    y_tags = pd.read_pickle('y_tags.pkl')

    clean_array = CleanedInputLemma(input)
    temp = []
    strin = [s + " " for s in clean_array]
    temp.append(''.join(strin))
    clean_input = temp

    bi_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    print("input is : ", input)
    print("clean array is : ", clean_array)
    print("clean input is : ", clean_input)

    emb_data = pd.read_pickle('./embData.pkl')
    featData = pd.read_pickle('./featData.pkl')

    clean_emb_array_clean = emb_data["Clean_Input"].values.tolist()
    temp = []
    for strin in clean_emb_array_clean:
        strin = [s + " " for s in strin]
        temp.append(''.join(strin))
    clean_emb_array_clean = temp

    clean_arr = featData["Clean_Input"].values.tolist()
    temp = []
    for strin in clean_arr:
        strin = [s + " " for s in strin]
        temp.append(''.join(strin))
    clean_arr = temp

    bi_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    bi_vect = bi_vectorizer.fit_transform(clean_arr)
    bi_vecto = bi_vectorizer.transform(clean_input)

    'Bigram input is *%s*' % bi_vecto

    model = gensim.models.Word2Vec(emb_data["SemiClean_Input"].tolist(), size=150, window = 10, min_count = 2, workers = 10)  # an empty model, no training yet
    model.train(clean_emb_array_clean, total_examples = len(clean_emb_array_clean), epochs = 10)  # can be a non-repeatable, 1-pass generator

    model.wv.most_similar(positive="json")

    X_average = word_averaging_list(model.wv,clean_arr)
    X_predict = word_averaging_list(model.wv,clean_input)

    'Embedding input is *%s*' % X_predict

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=1000,
                                   stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(clean_arr)
    tfidf_clean = tfidf_vectorizer.transform(clean_input)

    'tfidf is *%s*' % tfidf

"np shapes bi_vecto are : ", np.shape(bi_vecto.toarray())
print("bi vecto is : ", bi_vecto.toarray())
"np shapes are : ", np.shape(X_predict)
"np shapes tfidf are : ", np.shape(tfidf.toarray())
#
'''
## Random Forest
'''
# bi grammes
# rdf_clf = joblib.load('./RDF_fit.txt')
# predict_rdf = rdf_clf.predict(bi_vecto.toarray())
#
# "prediction is ", predict_rdf
# "Predicted tags are : "
# np.where(predict_rdf[0] == 1)[0]
# for values in np.where(predict_rdf[0] == 1)[0]:
#     st.text(y_tags.iloc[values].values[0])


'''
## k-Neighbors
'''
# embedding
neigh = joblib.load('./kN_fit.txt')
predict_neigh = neigh.predict(X_predict)

"prediction is ", predict_neigh
"Predicted tags are : "
np.where(predict_neigh[0] == 1)[0]
for values in np.where(predict_neigh[0] == 1)[0]:
    st.text(y_tags.iloc[values].values[0])

'''
## MLP
'''
# tfidf
mlp_clf = joblib.load('./MLP_fit 2.txt')
result_mlp = mlp_clf.predict(tfidf_clean.toarray())

"prediction is ", result_mlp
"Predicted tags are : "
tfidf_clean.toarray()
np.where(result_mlp[0] == 1)[0]
for values in np.where(result_mlp[0] == 1)[0]:
    st.text(y_tags.iloc[values].values[0])
