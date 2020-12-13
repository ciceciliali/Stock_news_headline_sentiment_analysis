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

def split_by_news(df_1):
	new_list = []
	my_list = df_1.values.tolist()
	#print(type(df_1['News'].tolist()))
	for rowID, date, p_change, headlines in my_list:
		meow = headlines.split('|')
		for headline in meow:
			cur_row = (date, p_change, headline)
			new_list.append(cur_row)
	return new_list

def write_csv(file_path, data):
	with open(file_path, 'w') as out_csv:
		fieldnames = ['date','change','headline']
		csv_wr = csv.writer(out_csv, delimiter=',')
		csv_wr.writerow(fieldnames)
		csv_wr.writerows(data)

def main():
	input_file = ('nandropped_with_news.csv')
	df_1 = pd.read_csv(input_file, sep=',', engine='python')
	splited = split_by_news(df_1)
	file_path = 'ind_news.csv'
	write_csv(file_path,splited)

	#df['change'] = df.apply(lambda row: find_change(row), axis=1)
	#df = df.drop(df.columns[[1, 2]], axis=1)
	#df.to_csv('change.csv')

	labeled = ('labeled.csv')
	df_2 = pd.read_csv(labeled, sep=',', engine='python')
	df_2 = df_2.drop(df_2.columns[0], axis=1)
	df_2 = df_2.dropna()
	#print(df_2)
	df_2.to_csv('gold_lb_test.csv')
	# df_2 = pd.read_csv(input_file, sep=',', engine='python')
	# df_2 = df_2.drop(df_2.columns[[0, 2]], axis=1)
	# news_dict = get_news_dict(df_2) 

	# df_1['News'] = df_1.apply(lambda row: fill_news(row, news_dict), axis=1)
	# #df_1 = df_1.drop(df.columns[[0, 2]], axis=1)
	# df_1 = df_1.dropna()

	# df_1.to_csv('nandropped_with_news.csv')

	
	
if __name__ == '__main__':
	main()