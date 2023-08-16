import nltk
import numpy as np
import pandas as pd
from keras import models
from keras import layers
from tensorflow.keras.layers import Embedding
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import seaborn as sns
import  time
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown
data = list(nltk.corpus.brown.tagged_sents(tagset='universal'))
import multiprocessing
from gensim.models import Word2Vec
word_vectors = Word2Vec(sentences=brown.sents(), size=100, window=5, min_count=1, workers=4).wv
def get_tag_vocabulary(data):
    tag2id = []
    words2id=[]
    for sent in data:
      words_sent_list=[]
      tags_sent_list=[]
      for word,tag in sent:
          words_sent_list.append(word)
          tags_sent_list.append(tag)
      tag2id.append(tags_sent_list)
      words2id.append(words_sent_list)
    return tag2id,words2id


tag2id,words2id = get_tag_vocabulary(data)
word_tokenizer = Tokenizer(lower = 'True')                      
word_tokenizer.fit_on_texts(words2id)                    
words_id = word_tokenizer.texts_to_sequences(words2id)
words_list = pad_sequences(words_id, maxlen=180, padding="pre", truncating="post")


tag_tokenizer = Tokenizer(lower = 'True')                      
tag_tokenizer.fit_on_texts(tag2id)                    
tags_id = tag_tokenizer.texts_to_sequences(tag2id)
tags_list= pad_sequences(tags_id, maxlen=180, padding="pre", truncating="post")

import tensorflow as tf
def sen_vec(sent):
    words=sent.split(" ")
    print(words)
    l = []
    print(words[1])
    for i in range(len(words)):
        print(words[i])
        x=0
        for j in range(len(words_list)):
            for k in range(len(words_list[j])):
                print(words2id[j][1])
                # print(words[i])
                if (x==0 and words2id[j][k]==words[i]):
                    l.append(int(words_list[j][k]))
                    x=1
                    break
            if(x==1):
                break
    t=tf.convert_to_tensor(l)
    t=tf.reshape(t, [1, t.shape[0]])
    y = pad_sequences(t, maxlen=180, padding="pre", truncating="post")
    return y

model = Word2Vec(words2id , size = 200 , window = 5, workers=4 , min_count =1)
vocab_size= list(model.wv.vocab)
word2id = word_tokenizer.word_index
model.wv.save_word2vec_format("word2vec.txt", binary = False)
embedding_weights = np.zeros((len(vocab_size), 200))
e={}
f=open("word2vec.txt",encoding = "utf-8")
for l in f:
    values = l.split()
    words = values[0]
    c = np.asarray(values[1:])
    e[words] = c
f.close()
k=0
for w , i in word2id.items():
    # if k == 0:
    #     print(i)
    #     print(e[w])
    #     k=1
    try:
        embedding_weights[i,:] = e[w]
    except KeyError:
        pass    

tags_list = to_categorical(tags_list)

len_test = len(tags_list)/5
ls=[]
a=[]
# print(len_test*5)
for i in range(5):
    k=int(i*len_test)
    l=int((i+1)*len_test)
    X_train = np.concatenate((words_list[:k],words_list[l:]),axis=0)
    Y_train= np.concatenate((tags_list[:k],tags_list[l:]),axis=0)
    X_test = words_list[k:l]
    Y_test = tags_list[k:l]
    model = models.Sequential()
    print(i)
    e_l=Embedding(input_dim = embedding_weights.shape[0],output_dim = embedding_weights.shape[1],weights = [embedding_weights],input_length = 180 , trainable=True)
    model.add(e_l)
    model.add(layers.Dense(units=100, activation='relu'))
    model.add(layers.Dense(units=13, activation='softmax'))
    model.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['acc',f1_m,precision_m, recall_m])
    training = model.fit(X_train,Y_train,batch_size=128,epochs=1,validation_data=(X_test, Y_test))
    # loss , accuracy = model.evaluate(X_test,Y_test)
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test, verbose=0)
    ls.append(loss)
    a.append(accuracy)
    f05_score = (1.25)*((precision*recall)/((0.25*precision)+recall+K.epsilon()))
    f2_score = 5*((precision*recall)/((4*precision)+recall+K.epsilon()))
    # predictions
    # predicted_classes = numpy.argmax(predictions, axis=1)
    # confusion_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(y_pred))
    print(loss , accuracy , f1_score,precision,recall,f2_score,f05_score)
l=model.layers[-1].output
x=model.layers[-1].get_weights()
key_given_val(e,x)
# x=list(x)
# x=np.ndarray.tolist(x)
# e_word = list(e.keys())
# print(e.keys())
# print(x)
# e_val = list()
# print e.keys()[e.values().index(x[1])]

def key_given_val(e,val):
    # for i in 
    x=[]
    for k,v in e.items():
        print(val)
        if val in v:
            print(val)
            x.append[k]
    return x
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



