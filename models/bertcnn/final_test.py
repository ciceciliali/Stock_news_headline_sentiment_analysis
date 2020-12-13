
# downloading BERT module 
# pip install bert-for-tf2
# pip install sentencepiece

import os
import re
import random
import numpy as np 
import pandas as pd 

import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# ML Modelling Libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, precision_score,recall_score,roc_auc_score
from sklearn.metrics import accuracy_score, plot_precision_recall_curve


# BERT and Deep Learning Libs
import bert
import tensorflow_hub as hub
import tensorflow as tf
from keras import layers
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

import nltk
nltk.download('stopwords')

import warnings
warnings.filterwarnings("ignore")

ner = pd.read_csv('ner.csv', 
                  encoding= 'ISO-8859-1',
                  error_bad_lines=False)
ner_dataset = pd.read_csv('ner_dataset.csv',
                  encoding='ISO-8859-1',
                  error_bad_lines=False)
stock_data = pd.read_csv('stock_data.csv',
                  encoding='ISO-8859-1',
                  error_bad_lines=False)

ner.columns

ner = ner[['prev-word','prev-pos','word','pos','next-word','next-pos','tag']]
ner.head(10)

ner.pos.unique()

ner.fillna("None",inplace=True)
ner.tag.unique()

# no cleaning function is required to be passed since we are training model on entire corpus

lb_tag = LabelEncoder().fit(ner.tag)
ner.tag = lb_tag.fit_transform(ner.tag)

lb_pos= LabelEncoder().fit(ner.pos)
ner.pos = lb_pos.fit_transform(ner.pos)

ner['prev-pos'] = lb_pos.fit_transform(ner['prev-pos'])
ner['next-pos'] = lb_pos.fit_transform(ner['next-pos'])

"""Lets import the pretrained BERT layer from TF HUB to harness its tokenizer"""

# import BERT layer from Tensorflow hub URL
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

# Using BERT's inbuilt tokenizer

tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

def vectorize(series):
    # Converting words into word vectors to feed into Model using BERT Tokenizer
    series.fillna("None", inplace = True)
    series = series.apply(lambda word : tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
    # Tokenizer returns list hence extracting numbers from it
    return series.map(lambda x: 0 if len(x) == 0 else x[0])

ner.word = vectorize(ner.word)
ner['prev-word'] = vectorize(ner['prev-word'])
ner['next-word'] = vectorize(ner['next-word'])

num_classes = ner.tag.nunique()
num_classes

X = ner.drop(columns='tag').values
y = ner['tag'].values

print(X.shape)
y = y.reshape(y.shape[0])
print(y.shape)

# Splitting into Train/Test sets

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.20)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# changing labels of stock data to 0 and 1
stock_data['Sentiment'] = stock_data['Sentiment'].apply(lambda x: 0 if x == -1 else 1)

"""Preprocessing functions"""

# clean text - lemmitization and removing stop-words, URLs, punctuations and special chars


stopwords = stopwords.words('english') # from nltk module
def preprocess(sentence):
    
    result = []
    
    s = BeautifulSoup(sentence, "lxml").get_text()
    
    # Removing the URL links
    s = re.sub(r"https?://[A-Za-z0-9./]+", ' ', s)
    
    # Keeping only letters
    s = re.sub(r"[^a-zA-Z.!?']", ' ', s)
    
    # Removing additional whitespaces
    s = re.sub(r" +", ' ', s)
    
    token_list = tokenizer.tokenize(s)
    
    for token in token_list:
        if (token not in list(string.punctuation))and(token not in stopwords):
            result.append(token)
        else:
            continue
    
    return result

# Adding Classification and Separator token for each sentence -- BERT input format

def add_std_tokens(token_list):
    return ["[CLS]"] + token_list + ["[SEP]"]

# FUNCTION 1: TO GET WORD VECTOR FROM A LIST OF TOKENS

def get_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)


# FUNTION 2: TO GET WHETHER OUR TOKENS HAVE [PAD] PADDING OR NOT
# NOTE: In this case it is not important but we will use it to maintain general norm

def get_masks(tokens):
    return np.char.not_equal(tokens, '[PAD]').astype(int)


# FUNCTION 3 : TO GET ID's OF SEGMENTATION TOKENS

def get_segs(tokens):
    curr_seg_id=0
    seg_ids =[]
    for tok in tokens:
        seg_ids.append(curr_seg_id)
        if tok=="[SEP]":
            curr_seg_id = 1- curr_seg_id
            
            # 1 becomes 0 and 0 becomes 1
            # 1 denoting [SEP] token and 0 any other token
            
    return seg_ids

"""Creating Dataset with Appropriate Format for for BERT layer.
Applying the three functions on the shuffled and sorted data in format : - 

( [wordvecs] , [pads] , [seps] , labels )

NOTE : keras.preprocessing.sequence.pad_sequences cannot be used because it doesnt support string and int.
"""

# Applying text cleaning and merging labels and lengths for sorting

labels = stock_data.Sentiment.values
cleaned_data = [add_std_tokens(preprocess(sent)) for sent in stock_data.Text]


data_with_len = [[sent, labels[i], len(sent)]
                 for i, sent in enumerate(cleaned_data)]

# Shuffle and Sort the dataset

random.shuffle(data_with_len)

data_with_len.sort(key=lambda x: x[2])

# Applying the 3 functions to get input in appropriate format

compiled_data = [([ get_ids(sent_idx[0]), list(get_masks(sent_idx[0])), get_segs(sent_idx[0])],
                    sent_idx[1]) for sent_idx in data_with_len]

"""Using tf.data.Dataset module to make a padded Dataset for BERT Layer"""

batch_size = 32
num_batches = len(compiled_data) // batch_size #180
num_test_batches = num_batches // 15           #12

# making tf.data.Dataset generator objects
dataset_gen = tf.data.Dataset.from_generator(lambda : compiled_data, 
                                             output_types=(tf.int32,tf.int32))

# using the padded batch function to make a batch generator
batch_gen = dataset_gen.padded_batch(batch_size, 
                                     padded_shapes=((3,None),()), 
                                     padding_values = (0,0))

# using the shuffle attribute
batch_gen.shuffle(num_batches)

# getting batched tensor datasets from generator
train_data = batch_gen.skip(num_test_batches)
test_data = batch_gen.take(num_test_batches)

class DCNN(tf.keras.Model):
    
    # making a constructor for default params
    def __init__(self,
                 FC_units =512,
                 num_filters=32,
                 num_classes=2,
                 droupout = 0.4,
                 name = "DCNN"):
        
        # calling superclass constructor
        super(DCNN,self).__init__(name=name)
        
        # adding layers to DCNN model object
        
        self.bert_layer = hub.KerasLayer(
                            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
        
        self.bigram_layer = layers.Conv1D(
                                filters=num_filters,
                                kernel_size=2,
                                padding='valid',
                                activation ='relu')
        
        self.trigram_layer = layers.Conv1D(
                                filters=num_filters,
                                kernel_size=3,
                                padding='valid',
                                activation ='relu')
        
        self.fourgram_layer = layers.Conv1D(
                                filters=num_filters,
                                kernel_size=4,
                                padding='valid',
                                activation ='relu')
        
        self.batchnorm = layers.BatchNormalization()
        self.layernorm = layers.LayerNormalization()
        self.pool_layer = layers.GlobalMaxPool1D()
        self.dense_layer = layers.Dense(FC_units,activation='relu')
        self.dropout_layer = layers.Dropout(rate=dropout_rate)
        
        if num_classes == 2:
            self.output_layer = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.output_layer = layers.Dense(units=nb_classes,
                                           activation="softmax")
            
    # Embed Tensors into BERT Layer, embs gives output
    def embed_with_bert(self, all_tokens):
            
        _, embs = self.bert_layer([all_tokens[:, 0, :],
                                   all_tokens[:, 1, :],
                                   all_tokens[:, 2, :]])
        return embs

        
    # Implement the Architecture in the call function
    def call(self, inputs, training):
            
        x = self.embed_with_bert(inputs)
            
        bigram = self.bigram_layer(x)
        bigram = self.layernorm(bigram)
        bigram = self.batchnorm(bigram)
        bigram = self.pool_layer(bigram)
            
        trigram = self.trigram_layer(x)
        trigram = self.layernorm(trigram)
        trigram = self.batchnorm(trigram)
        trigram = self.pool_layer(trigram)
        
        fourgram = self.fourgram_layer(x)
        fourgram = self.layernorm(fourgram)
        fourgram = self.batchnorm(fourgram)
        fourgram = self.pool_layer(fourgram)
        
        merged = tf.concat([bigram, trigram, fourgram],axis=-1) 
        # (batch_size, 4 * num_filters)
        merged = self.dense_layer(merged)
        merged = self.dropout_layer(merged)
        output = self.output_layer(merged)
            
        return output

from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy
opt = Adam()

# Callback to prevent overfit

early_stopping_callback =  EarlyStopping(monitor = 'val_accuracy',
                                         min_delta = 0.01,
                                         patience = 6,
                                         restore_best_weights=True)

FC_units = 64
num_filters = 4
num_classes = 2
dropout_rate = 0.2
batch_size = 32
num_epochs = 4

# Making model
model = DCNN(FC_units = FC_units,
             num_filters=num_filters,
             num_classes=num_classes,
             droupout = dropout_rate)

# Compiling
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

"""Training model """

num_train_batches = num_batches - num_test_batches
num_valid_batches = num_train_batches // 5


x_train=train_data.skip(num_valid_batches) 
x_valid=test_data.take(num_valid_batches)

# Fitting data using crossvalidation
history = model.fit(x_train, 
                    epochs=num_epochs,
                    validation_data = x_valid,
                    callbacks =[early_stopping_callback])

results = model.evaluate(test_data)
results

y_pred = model.predict(test_data)
y_pred.shape

print(test_data)

"""Extracting values from nested Tensors in tf.data.Dataset objects"""

lis=[]
y = tfds.as_numpy(test_data)
for i,j in enumerate(y):
    tensors,labels = j
    lis.extend(labels)
    
y_true = np.array(lis,dtype='float32')
y_true = y_true.reshape(y_pred.shape[0],)
y_pred = y_pred.reshape(y_pred.shape[0],)

print("ROC AUC score : ", roc_auc_score(y_true,y_pred,average="micro"))

# making predictions discrete
y_pred=y_pred>0.5
y_pred=y_pred.astype(int)

print("f1_score: ", f1_score(y_true,y_pred,average="micro"))
print("precision_score: ", precision_score(y_true,y_pred,average="macro"))
print("recall_score: ", recall_score(y_true,y_pred,average="macro"))