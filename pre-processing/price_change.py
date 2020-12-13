import os
import re
import spacy
import pandas as pd
import numpy as np
import math
import emoji
import string
import contractions
from math import isnan
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.util import ngrams

def find_change(row):
	open_price = row['Open']
	close_price = row['Close']
	return_value = 0
	change = ((close_price - open_price)/open_price)*100
	if change > 0.5:
		return_value = 1
	if change < -0.5:
		return_value = -1
	return return_value


def main():
	input_file = ('AAPL.csv')
	df = pd.read_csv(input_file, sep=',', engine='python')
	df = df.drop(df.columns[[2, 3, 5, 6]], axis=1)
	df['change'] = df.apply(lambda row: find_change(row), axis=1)
	df = df.drop(df.columns[[1, 2]], axis=1)
	df.to_csv('change.csv')
	
	
if __name__ == '__main__':
	main()