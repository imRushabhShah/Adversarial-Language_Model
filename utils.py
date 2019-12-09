import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import math
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation,TimeDistributed
from keras.layers import Bidirectional, LSTM
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate,BatchNormalization,MaxPooling1D, Convolution1D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, RepeatVector, Permute, merge
from keras import backend as K
from keras import layers
from keras.models import Sequential, models
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    # So we only measure F1 on the target y value:
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
	
def error_conf(error, n):
    term = 1.96*sqrt((error*(1-error))/n)
    lb = error - term
    ub = error + term    
    return lb, ub

# define a function that allows us to evaluate our models	
def evaluate_model(predict_fun, X_train, y_train, X_test, y_test):
    '''
    evaluate the model, both training and testing errors are reported
    '''
    # training error
    y_predict_train = predict_fun(X_train)
    train_acc = accuracy_score(y_train,y_predict_train)
    
    # testing error
    y_predict_test = predict_fun(X_test)
    test_acc = accuracy_score(y_test,y_predict_test)
    
    return train_acc, test_acc

# read in our data and preprocess it
def read_csv(TEXT_DATA=TEXT_DATA):
    df = pd.read_csv(TEXT_DATA)
    df.drop(labels=['id','title'], axis='columns', inplace=True)
    # only select stories with lengths gt 0 -- there are some texts with len = 0
    mask = list(df['text'].apply(lambda x: len(x) > 0))
    df = df[mask]
    return df

# prepare text samples and their labels
def getTextLabels(df=df):
    texts = df['text']
    labels = df['label']
    print('Found %s texts.' %texts.shape[0])
    return texts, labels

# vectorize the text samples into a 2D integer tensor 
def getEmbedding(texts=texts,EMBEDDING_FILE=EMBEDDING_FILE, EMBEDDING_DIM=EMBEDDING_DIM,
    MAX_NUM_WORDS=MAX_NUM_WORDS):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    data = pad_sequences(sequences, 
                         maxlen=MAX_SEQUENCE_LENGTH, 
                         padding='pre', 
                         truncating='pre')

    print('Found %s unique tokens.' % len(word_index))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    # Read the glove word vectors (space delimited strings) into a dictionary from word->vector.
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf8"))

    '''
    Use these vectors to create our embedding matrix, with random initialization for words that aren't 
    in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init
    '''
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    emb_mean,emb_std
    word_index = tokenizer.word_index
    nb_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return tokenizer, embedding_matrix

def split_data(data=data,labels=labels, TEST_SPLIT=TEST_SPLIT):
    return train_test_split(data, labels.apply(lambda x: 0 if x == 'FAKE' else 1), test_size=TEST_SPLIT)



