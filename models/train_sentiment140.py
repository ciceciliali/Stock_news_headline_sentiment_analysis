import csv
import os
import re
import pandas as pd
import nltk
from tqdm.auto import tqdm
import contractions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

'''
TO RUN CODE:
Everything should just runs in main()
F1 score are printed for all models.
To print the accuracy for a model, check line 108-185.
Comment out or uncomment which prediction to calculate accuracy for. 
'''


def load_data_sentiment140():
    # file_train = os.path.join('Data', 'Sentiment140', 'training.1600000.processed.noemoticon.csv')
    # n = 100
    # num_lines = 1600000
    # skip_idx = [x for x in range(num_lines) if x % n != 0]
    # df = pd.read_csv(file_train, sep=',', engine='python', skiprows=skip_idx)
    # df.columns = ['Sentiment', 'ID', 'Date', 'Query', 'User', 'Text']
    # df = df[['Sentiment', 'Text']]
    # df = df[df['Sentiment'].isin([4, 0])]

    file_train = os.path.join('Data', 'sentiment140_filtered.csv')
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
    df_senti = load_data_sentiment140()
    test_df = load_test_data()

    ''' Sentiment140 testing'''
    cleaned_text = clean_text(df_senti)
    sentiment = df_senti['Sentiment'].tolist()

    # turn labels into indexes
    rel_index = []
    for rel in sentiment:
        if rel == 0:
            rel_index.append(0)
        if rel == 4:
            rel_index.append(1)

    test_data = test_df['HEADLINE'].tolist()
    check_test = test_df['SENTIMENT'].tolist()

    # split data 80% for training and 20% for validation
    x_train, x_test, y_train, y_test = train_test_split(cleaned_text, rel_index, test_size=0.2,
                                                        random_state=88)

    # use scikit learn tfidf vectorizer on the text, limit features to 1000 to minimize over-fitting
    tfidf_vec = TfidfVectorizer()
    x_train = tfidf_vec.fit_transform(x_train)
    x_test = tfidf_vec.transform(x_test)

    clf = MultinomialNB(alpha=0.1).fit(x_train, y_train)
    clf2 = LogisticRegression(max_iter=100).fit(x_train, y_train)
    clf3 = SVC(kernel='rbf', gamma=1, C=2, tol=0.1).fit(x_train, y_train)
    clf4 = LinearSVC(loss='hinge', tol=0.1).fit(x_train, y_train)
    clf5 = DecisionTreeClassifier().fit(x_train, y_train)
    clf6 = RandomForestClassifier().fit(x_train, y_train)

    predictions = clf.predict(x_test)
    predictions2 = clf2.predict(x_test)
    predictions3 = clf3.predict(x_test)
    predictions4 = clf4.predict(x_test)
    predictions5 = clf5.predict(x_test)
    predictions6 = clf6.predict(x_test)

    print('MultinomialNB F1 Score: [1] ', round(f1_score(y_test, predictions, average='micro'), 3))
    print('LogisticRegression F1 Score: [2] ', round(f1_score(y_test, predictions2, average='micro'), 3))
    print('SVC F1 Score: [3] ', round(f1_score(y_test, predictions3, average='micro'), 3))
    print('LinearSVC F1 Score: [4] ', round(f1_score(y_test, predictions4, average='micro'), 3))
    print('DecisionTreeClassifier F1 Score: [5] ', round(f1_score(y_test, predictions5, average='micro'), 3))
    print('RandomForestClassifier F1 Score: [6] ', round(f1_score(y_test, predictions6, average='micro'), 3))

    x_test = tfidf_vec.transform(test_data)

    #final_predictions = clf.predict(x_test)
    #final_predictions = clf2.predict(x_test)
    final_predictions = clf3.predict(x_test)
    #final_predictions = clf4.predict(x_test)
    #final_predictions = clf5.predict(x_test)
    #final_predictions = clf6.predict(x_test)

    output = []
    gold_output = []
    for x in range(len(final_predictions)):
        if final_predictions[x] == 1:
            output.append([test_data[x], 1])

        if final_predictions[x] == 0:
            output.append([test_data[x], -1])

        gold_output.append([test_data[x], check_test[x]])

    correct = 0
    total = len(output)
    for i in range(total):
        if output[i][1] == check_test[i]:
            correct += 1

    print('Using Sentiment140\nAcc:', correct / total)

    file_pred = os.path.join('Output', 'pred_140.csv')
    file_gold = os.path.join('Output', 'gold_140.csv')

    write_csv(file_pred, output)
    write_csv(file_gold, gold_output)


if __name__ == '__main__':
    main()
