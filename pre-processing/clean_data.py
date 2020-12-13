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
from collections import OrderedDict
from collections import Counter
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

def data_cleaning(df):
	info = [df['Sentiment'].values.tolist(),df['Text'].values.tolist()]
	df_data = list(zip(*info)) 
	clean_data = []
	sentiment = []
	pos = {}
	neg = {}
	counts = {}
	for i in range(0,len(df_data)):

		if int(df_data[i][0]) == 1:
			sentiment.append(1)
		if int(df_data[i][0]) == -1:
			sentiment.append(-1)
		#1) remove emails
		clean_sentence = re.sub(r'\s*\S*(@)\S*', '', str(df_data[i][1]))
		#remove mentions
		clean_sentence = re.sub(r"([@][\w_-]+)","",clean_sentence)
		#2) remove 10 digit phone numbers
		#clean_sentence = re.sub(r'\d{10}', '', clean_sentence)
		#3) remove $n
		clean_sentence = re.sub(r'\$[^ ]+', '', clean_sentence)
		#4) remove Times & dates 2/24 2:10pm, 6/30, 7:00 AM
		clean_sentence = re.sub(r'[0-9]*[:/][0-9]*\S*\s[A][M]|[0-9]*[:/][0-9]*\S*\s[P][M]|[0-9]*[:/][0-9]*\S*','',clean_sentence)
		#5) convert emojis
		clean_sentence = emoji.demojize(clean_sentence, delimiters = (' ',' '))
		#6) fix contractions
		clean_sentence = contractions.fix(clean_sentence)
		#7) remove links
		clean_sentence = re.sub(r'\s*\S*(http)\S*', '', clean_sentence)
		#8) keeping the hashtage info but removing the sign
		clean_sentence = clean_sentence.replace("#","")
		#9) all to lower case for easy tokenization and less features
		#clean_sentence = clean_sentence.lower()
		#10) remove &lt; &gt;
		clean_sentence = clean_sentence.replace("&lt;","")
		clean_sentence = clean_sentence.replace("&gt;","")
		#11) remove Punctuations
		clean_sentence = re.sub(r'[^A-Za-z0-9]+', ' ', clean_sentence)
		#12) lemmatize verbs
		tokenized = word_tokenize(clean_sentence)
		lemmatizer = WordNetLemmatizer()
		clean_tokens = []
		for word in tokenized:
			cur = lemmatizer.lemmatize(word, pos='v')
			clean_tokens.append(cur)
		#13) remove stop words
		stop_words = set(stopwords.words('english'))  
		filtered_sentence = [w for w in clean_tokens if not w in stop_words]  
		clean_data.append(filtered_sentence)
		for word in filtered_sentence:
			if sentiment[i] == 1:
				if word in pos and not word.isupper():
					pos[word] += 1
				else:
					pos[word] = 1
			else:
				if word in neg and not word.isupper():
					neg[word] += 1
				else:
					neg[word] = 1
			if word not in counts:
				counts[word] = 1
			else:
				counts[word] += 1
		neg = {key:val for key, val in neg.items() if val != 1}
		pos = {key:val for key, val in pos.items() if val != 1}
		alldata = {key:val for key, val in counts.items() if val != 1}
	#print(clean_data[5])
	#print(len(sentiment) == len(clean_data))

	f = open("pos.txt", "w")
	f.write(str(pos))
	f.close()

	f = open("neg.txt", "w")
	f.write(str(neg))
	f.close()

	f = open("counts.txt", "w")
	f.write(str(alldata))
	f.close()
	a_dictionary = dict(Counter(counts).most_common(50))
	keys = a_dictionary.keys()
	values = a_dictionary.values()
	plt.xticks(fontsize=6)
	plt.bar(keys, values, color = 'pink')
	plt.show()



	#check for duplicates 
	new_clean_data = []
	for i in range(0,len(clean_data)):
		cur_data = (' '.join(clean_data[i]),sentiment[i])
		if cur_data not in new_clean_data:
			#new_clean_data.append(clean_data[i])
			#new_senti.append(sentiment[i])
			new_clean_data.append(cur_data)

	#print(new_clean_data[:3])

	return new_clean_data

def half(df):
	pos = []
	neg = []
	final = []
	for rows in df:
		if str(rows[1]) == '-1':
			neg.append(rows[0])
		else:
			pos.append(rows[0])
	if len(pos) > len(neg):
		for i in range(0,len(neg)):
			cur_pos = (pos[i],1)
			cur_neg = (neg[i],-1)
			final.append(cur_pos)
			final.append(cur_neg)

	if len(neg) > len(pos):
		for i in range(0,len(pos)):
			cur_pos = (pos[i],1)
			cur_neg = (neg[i],-1)
			final.append(cur_pos)
			final.append(cur_neg)
	return final


def main():
	input_file = ('stock_data.csv')
	df = pd.read_csv(input_file, sep=',', engine='python')
	new_clean_data = data_cleaning(df)
	new_clean_data = half(new_clean_data)

	string = "Text,Sentiment" + "\n"
	for i in range(0,len(new_clean_data)):
		string = string + str(new_clean_data[i][0]) + "," + str(new_clean_data[i][1]) + '\n'
	f = open("clean_data.csv", "w")
	f.write(str(string))
	f.close()

if __name__ == '__main__':
	main()
