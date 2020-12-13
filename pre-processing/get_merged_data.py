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

def fill_news(row, news):
	if row['Date'] in news:
		return news[row['Date']]
	return None

def get_news_dict(df_2):
	list_df = df_2.values.tolist()
	d = {}
	for row in list_df:
		if row[1] not in d:
			d[row[1]] = []
		d[row[1]].append(row[0])
	for keys in d:
		d[keys] = '|'.join(d[keys])
	return d

def main():
	input_file = ('change.csv')
	df_1 = pd.read_csv(input_file, sep=',', engine='python')
	df_1 = df_1.drop(df_1.columns[[0]], axis=1)
	print(df_1)

	#df['change'] = df.apply(lambda row: find_change(row), axis=1)
	#df = df.drop(df.columns[[1, 2]], axis=1)
	#df.to_csv('change.csv')

	input_file = ('aapl_news.csv')
	df_2 = pd.read_csv(input_file, sep=',', engine='python')
	df_2 = df_2.drop(df_2.columns[[0, 2]], axis=1)
	news_dict = get_news_dict(df_2) 

	df_1['News'] = df_1.apply(lambda row: fill_news(row, news_dict), axis=1)
	#df_1 = df_1.drop(df.columns[[0, 2]], axis=1)
	df_1 = df_1.dropna()

	df_1.to_csv('nandropped_with_news.csv')

	
	
if __name__ == '__main__':
	main()