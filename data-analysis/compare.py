import os
import re
import spacy
import pandas as pd
import numpy as np
import math
import emoji
import string
import contractions
import csv
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


def compare(df_1,df_2):
	d_df1 = {}
	d_df2 = {}
	same = 0
	not_in = []
	for date,change,headline in df_1:
		d_df1[headline] = float(change)
	for hid,headline,sen,date in df_2:
		d_df2[headline] = float(sen)
	for key in d_df2:
		if key not in d_df1:
			not_in.append(key)
		if key in d_df1 and d_df2[key] == d_df1[key]:
			#print('here')
			same += 1
	print(not_in)
	print(same/len(d_df2))



def main():
	input_file = ('ind_news.csv')
	df_1 = pd.read_csv(input_file, sep=',', engine='python')
	df_1 = df_1.values.tolist()

	gold_lb_test = ('gold_lb_test.csv')
	df_2 = pd.read_csv(gold_lb_test, sep=',', engine='python')
	df_2 = df_2.values.tolist()

	compare(df_1,df_2)

	
	
if __name__ == '__main__':
	main()
