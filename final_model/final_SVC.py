import csv
import os
import re
import pandas as pd
import nltk
from tqdm.auto import tqdm
import contractions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score


def load_data():
    file_train = os.path.join('Data', 'stock_data.csv')
    df = pd.read_csv(file_train, sep=',', engine='python')

    return df


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
    df = load_data()
    test_df = load_test_data()

    sentiment = df['Sentiment'].tolist()
    cleaned_text = clean_text(df)

    test_data = test_df['HEADLINE'].tolist()
    check_test = test_df['SENTIMENT'].tolist()

    # turn labels into indexes
    rel_index = []
    for rel in sentiment:
        if rel == -1:
            rel_index.append(0)
        if rel == 1:
            rel_index.append(1)

    # split data 80% for training and 20% for validation
    x_train, x_test, y_train, y_test = train_test_split(cleaned_text, rel_index, test_size=0.2,
                                                        random_state=88)

    # use scikit learn tfidf vectorizer on the text
    tfidf_vec = TfidfVectorizer()
    x_train = tfidf_vec.fit_transform(x_train)
    x_test = tfidf_vec.transform(x_test)

    # SVC model
    clf = SVC(kernel='rbf', gamma=1, C=2, tol=0.1).fit(x_train, y_train)
    # predict on dev set
    predictions = clf.predict(x_test)
    # f1 score on dev set
    print('SVC F1 Score: ', round(f1_score(y_test, predictions, average='micro'), 3))

    # predict on test set
    x_test = tfidf_vec.transform(test_data)
    predictions = clf.predict(x_test)

    final = []
    output = []
    gold_output = []
    for x in range(len(predictions)):
        if predictions[x] == 1:
            final.append(1)
            output.append([test_data[x], 1])

        if predictions[x] == 0:
            final.append(-1)
            output.append([test_data[x], -1])

        gold_output.append([test_data[x], check_test[x]])

    # calculate the accuracy of the prediction
    correct = 0
    total = len(output)
    for i in range(total):
        if output[i][1] == check_test[i]:
            correct += 1

    print('SVC Acc:', correct / total)

    # output to csv file
    file_pred = os.path.join('Output', 'pred_SVC.csv')
    file_gold = os.path.join('Output', 'gold_SVC.csv')
    write_csv(file_pred, output)
    write_csv(file_gold, gold_output)


if __name__ == '__main__':
    main()
