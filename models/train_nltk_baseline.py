import csv
import os
import re
import pandas as pd
from tqdm.auto import tqdm
import contractions
from sklearn.metrics import f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk

'''
TO RUN CODE: JUST RUN IT
If NLTK Vader is not installed, uncomment line 109 to download
'''

def load_test_data():
    # load test data
    file_train = os.path.join('Data', 'labeled.csv')
    df = pd.read_csv(file_train, sep=',', engine='python')

    df = df[df['SENTIMENT'].isin([1, -1])]

    df.drop_duplicates(subset=['HEADLINE'])

    # extract utterance
    df = df[['HEADLINE', 'SENTIMENT']]

    # make to list
    return df


def write_csv(file_path, data):
    with open(file_path, 'w') as out_csv:
        fieldnames = ['HEADLINE', 'SENTIMENT']
        csv_wr = csv.writer(out_csv, delimiter=',')
        csv_wr.writerow(fieldnames)
        csv_wr.writerows(data)


def lemmatize_text(sentence):
    # tokenize tweet
    tokenized_text = nltk.TweetTokenizer().tokenize(sentence)
    # verb lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    filtered_tokens = []
    for word in tokenized_text:
        token = lemmatizer.lemmatize(word)
        filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def remove_ticker(sentence):
    file_train = os.path.join('Data', 'nasdaq_ticker_list.csv')
    df = pd.read_csv(file_train, sep=',', engine='python')
    ticker = df['Symbol'].tolist()
    tokenized_text = sentence.split()
    filtered_tokens = []
    for word in tokenized_text:
        if word.upper() not in ticker:
            filtered_tokens.append(word)
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def clean_text(stock_data):
    text_list = stock_data['Text']
    cleaned_text = []
    for row in tqdm(range(len(text_list))):
        filtered_text = text_list[row]

        # make everything to lowercase
        filtered_text = filtered_text.lower()

        # remove emails
        filtered_text = re.sub(r'\S*@\S*', '', filtered_text)

        # remove HTML escaped char
        filtered_text = re.sub(r'&\S*', '', filtered_text)

        # remove URLs
        filtered_text = re.sub(r'\s*\S*(http)\S*', '', filtered_text)

        # remove phone numbers
        filtered_text = re.sub(r'\d{10}', '', filtered_text)

        # split contractions
        filtered_text = contractions.fix(filtered_text)

        # lemmatize text
        filtered_text = lemmatize_text(filtered_text)

        # Remove ticker
        #filtered_text = remove_ticker(filtered_text)

        cleaned_text.append(filtered_text)

    return cleaned_text


def main():
    test_df = load_test_data()

    test_data = test_df['HEADLINE'].tolist()
    check_test = test_df['SENTIMENT'].tolist()

    #nltk.download('vader_lexicon')
    sia = SIA()
    results = []

    for line in test_data:
        pol_score = sia.polarity_scores(line)
        results.append(pol_score)

    nltk_pred = []
    for line in results:
        if line['compound'] > 0:
            nltk_pred.append(1)
        else:
            nltk_pred.append(-1)

    print('NLTK F1 Score: [6] ', round(f1_score(check_test, nltk_pred, average='micro'), 3))

    final1 = nltk_pred
    correct = 0
    total = len(final1)
    for i in range(len(final1)):
        if final1[i] == check_test[i]:
            correct += 1

    print('Acc:', correct / total)

if __name__ == '__main__':
    main()
