import os
import spacy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import math
from itertools import chain
import gensim

#to run the MLP model, uncomment the model1 part in main, and comment out model2 part in main
#to run the CNN model, uncomment the model2 part in main, and comment out model1 part in main 
#this script uses model.txt from section for encoding, if run CNN, put model.txt in the same dir as this script

def load_test_data():
    # load test data
    file_train = os.path.join('labeled.csv')
    df = pd.read_csv(file_train, sep=',', engine='python')

    df = df[df['SENTIMENT'].isin([1, -1])]

    # extract utterance
    df = df[['HEADLINE', 'SENTIMENT']]

    # make to list
    return df

def get_all_relations(df_data):
    d = {}
    d_l = []
    for rows in df_data:
        tokens = str(rows[2]).split(" ")
        for token in tokens:
            if token not in d :
                d[token] = 1
                d_l.append(token)
            else:
                d[token] = d[token] + 1
    total = 0
    for keys in d:
        total = total + d[keys]
    return d,d_l

def get_all_features(df_data):
    d = {}
    d_l = []
    nlp = spacy.load('en_core_web_sm')
    all_stopwords = nlp.Defaults.stop_words
    for rows in df_data:
        info = nlp(rows[1])
        info = [word for word in info if not word in all_stopwords]
        for token in info:
            cur_token = token.text
            if cur_token not in d:
                d[cur_token] = 1
                d_l.append(cur_token)
            else :
                d[cur_token] = d[cur_token] + 1
    if "" in d:
        d.remove("")
    return d,d_l

def getX(df_data):
    d_l = []
    for rows in df_data:
        d_l.append(rows[1])

    return d_l

def gety(df_data, y_dict,all_relations_d):
    d_l = []
    for rows in df_data:
        relations = str(rows[2]).split(" ")
        cur_max = 0
        cur_relation = 'nan'
        for relation in relations:
            if all_relations_d[relation] > cur_max:
                cur_relation = relation
                cur_max = all_relations_d[relation]

        d_l.append(y_dict.index(cur_relation))
    tensor = torch.LongTensor(d_l)

    return tensor

def gety_dict(df_data):
    d = {}
    d_l = []
    count = 0
    for rows in df_data:
        if str(rows[1]) not in d:
            d[rows[1]] = count
            d_l.append(rows[1])
            count = count + 1
    return d,d_l 

class BBCNewsDataset(Dataset):
    # This class is an interface for our training/dev/test dataset
    # with pytorch
    def __init__(self, texts, labels, input_transformer):
        self.texts = texts
        self.labels = labels
        self.input_transformer = input_transformer

    def __getitem__(self, index): # Return a single example
    
        text = self.texts[index]
        label = self.labels[index]
        x = self.input_transformer(text) # Produces TF-IDF/List of indices from text
        
        return x, label

    def __len__(self):
        return len(self.texts)

class MultiLayerPerceptron(nn.Module):
    """
    At its simplest, a multilayer perceptron is a 2 layer network
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=False, dropout_p=0.2):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)

        self.add_dropout = dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        if self.add_dropout:
            logits = self.fc2(self.dropout(h))
        else:
            logits = self.fc2(h)

        #logits = F.softmax(logits,dim = -1)

        return logits

class MultiClassTrainer(object):
    """
    Trainer for training a multi-class classification model
    """

    def __init__(self, model, optimizer, loss_fn, device="cpu", log_every_n=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        
        self.log_every_n = log_every_n if log_every_n else 0


    def _print_summary(self):
        print(self.model)
        print(self.optimizer)
        print(self.loss_fn)

    def train(self, loader):
        """
        Run a single epoch of training
        """

        self.model.train() # Run model in training mode

        loss_history = []
        running_loss = 0.
        running_loss_history = []

        for i, batch in tqdm(enumerate(loader)):
            batch_size = batch[0].shape[0]
            self.optimizer.zero_grad() # Always set gradient to 0 before computing it

            logits = self.model(batch[0].to(self.device)) # __call__ model() in this case: __call__ internally calls forward()
            # [batch_size, num_classes]

            loss = self.loss_fn(logits, batch[1].view(-1).to(self.device)) # Compute loss: Cross entropy loss

            loss_history.append(loss.item())

            

            running_loss += (loss_history[-1] - running_loss) / (i + 1) # Compute rolling average

            if self.log_every_n and i % self.log_every_n == 0:
                print("Running loss: ", running_loss)

            running_loss_history.append(running_loss)

            loss.backward() # Perform backprop, which will compute dL/dw

            nn.utils.clip_grad_norm_(self.model.parameters(), 3.0) # We clip gradient's norm to 3

            self.optimizer.step() # Update step: w = w - eta * dL / dW : eta = 1e-2 (0.01), gradient = 5e30; update value of 5e28

        print("Epoch completed!")
        print("Epoch Loss: ", running_loss)
        print("Epoch Perplexity: ", math.exp(running_loss))

        # The history information can allow us to draw a loss plot
        return loss_history, running_loss_history

    def evaluate(self, loader, labels):
        """
        Evaluate the model on a validation set
        """

        self.model.eval() # Run model in eval mode (disables dropout layer)

        batch_wise_true_labels = []
        batch_wise_predictions = []

        loss_history = []
        running_loss = 0.
        running_loss_history = []

        with torch.no_grad(): # Disable gradient computation - required only during training
            for i, batch in tqdm(enumerate(loader)):
                # batch[0] shape: (batch_size, input_size)

                logits = self.model(batch[0].to(self.device)) # Run forward pass (except we don't store gradients)
                # logits shape: (batch_size, num_classes)
                
                loss = self.loss_fn(logits, batch[1].view(-1).to(self.device)) # Compute loss
                # No backprop is done during validation
                
                # Instead of using CrossEntropyLoss, you use BCEWithLogitsLoss
                # BCEWithLogitsLoss - independently calculates loss for each class
                

                loss_history.append(loss.item())

                running_loss += (loss_history[-1] - running_loss) / (i + 1) # Compute rolling average
                
                running_loss_history.append(running_loss)

                # logits : [batch_size, num_classes] and each of the values in logits can be anything (-infinity, +infity)
                # Converts the raw outputs into probabilities for each class using softmax
                probs = F.softmax(logits, dim=-1) 
                # probs shape: (batch_size, num_classes)
                # -1 dimension picks the last dimension in the shape of the tensor, in this case 'num_classes'
                

                # softmax vector: [[0.1, 0.2, 0.6, 0.1, 0.0], [0.9, 0.01, 0.01, 0.01, 0.07]]
                # output tensor: [2, 0]
                #print(torch.argmax(probs))
                #print(probs)
                predictions = torch.argmax(probs, dim=-1) # Output predictions; Argmax picks the index with the highest probability among all the classes (choosing our most probable class)
                # predictions shape: (batch_size)
                #print(predictions)

                batch_wise_true_labels.append(batch[1].tolist())
                batch_wise_predictions.append(predictions.tolist())
        
        # flatten the list of predictions using itertools
        all_true_labels = list(chain.from_iterable(batch_wise_true_labels))
        all_predictions = list(chain.from_iterable(batch_wise_predictions))

        # Now we can generate a classification report
        #print("Classification report after epoch:")
        #print(all_predictions)
        #print(all_true_labels)
        correct = 0
        for i in range(0,len(all_predictions)):
            if all_predictions[i] == all_true_labels [i]:
                correct = correct + 1
        print("accuracy: ", correct/len(all_predictions))

        return loss_history, running_loss_history

    def get_model_dict(self):
        return self.model.state_dict()

    def run_training(self, train_loader, valid_loader, labels, n_epochs=10):
        # Useful for us to review what experiment we're running
        # Normally, you'd want to save this to a file
        self._print_summary()

        train_losses = []
        train_running_losses = []

        valid_losses = []
        valid_running_losses = []

        for i in range(n_epochs):
            loss_history, running_loss_history = self.train(train_loader)
            valid_loss_history, valid_running_loss_history = self.evaluate(valid_loader, labels)

            train_losses.append(loss_history)
            train_running_losses.append(running_loss_history)

            valid_losses.append(valid_loss_history)
            valid_running_losses.append(valid_running_loss_history)

        # Training done, let's look at the loss curves
        all_train_losses = list(chain.from_iterable(train_losses))
        all_train_running_losses = list(chain.from_iterable(train_running_losses))

        all_valid_losses = list(chain.from_iterable(valid_losses))
        all_valid_running_losses = list(chain.from_iterable(valid_running_losses))

        train_epoch_idx = range(len(all_train_losses))
        valid_epoch_idx = range(len(all_valid_losses))
        # sns.lineplot(epoch_idx, all_losses)
        #sns.lineplot(train_epoch_idx, all_train_running_losses)
        #sns.lineplot(valid_epoch_idx, all_valid_running_losses)
        #plt.show()

    def run_testing(self, test_loader):
        self.model.eval() # Run model in eval mode (disables dropout layer)

        batch_wise_true_labels = []
        batch_wise_predictions = []

        with torch.no_grad(): # Disable gradient computation - required only during training
            for i, batch in tqdm(enumerate(test_loader)):
                # batch[0] shape: (batch_size, input_size)

                logits = self.model(batch[0].to(self.device)) # Run forward pass (except we don't store gradients)
                # logits shape: (batch_size, num_classes)
                
                probs = F.softmax(logits, dim=-1) 
        
                #print(torch.argmax(probs))
                predictions = torch.argmax(probs, dim=-1) # Output predictions; Argmax picks the index with the highest probability among all the classes (choosing our most probable class)
                #print(predictions)

                batch_wise_predictions.append(predictions.tolist())

        all_predictions = list(chain.from_iterable(batch_wise_predictions))
        return all_predictions

class TextConvolver(nn.Module):
    """
    This model is based on "Convolutional Neural Networks for Sentence Classification" 
    described here (Kim 2014):
    https://arxiv.org/pdf/1408.5882.pdf

    I modified the implemenation defined here:
    https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py

    The contributors use 2D Cnns with kernel size (1 x K), but we can replace
    those with 1D convolutions with kernel size K.

    Additionally, we use pretrained word embeddings (English Wikipedia skip-gram)
    for our prediction, unlike the original paper which uses Word2Vec 
    100 billion words.
    """
    def __init__(self, input_size, output_size, kernel_sizes, channel_size=32, dropout=False, dropout_p=0.1, w2v_weights=None):
        super(TextConvolver, self).__init__()

        # Embedding: takes an integer index representing the word, and returns the corresponding vector for that word
        self.embedding = nn.Embedding.from_pretrained(w2v_weights)
        # self.embedding = nn.Embedding(vocab_size, embedding_dim) - nn.Embedding(314815, 300)
        embed_dim = 300

        self.frozen_embedding = nn.Embedding.from_pretrained(w2v_weights)
        self.frozen_embedding.requires_grad = False # Disable gradient updates, freezes the parameters from updates during training

        # Input
        # Perform convolution over it
        # Pass it through an activation
        # 

        self.add_dropout = dropout
        self.dropout = nn.Dropout(dropout_p)

        # Define an iterable set of parallel layers which are given the same input
        self.convs = nn.ModuleList([nn.Conv1d(2 * embed_dim, channel_size, kernel_size) for kernel_size in kernel_sizes])

        self.fc = nn.Linear(len(kernel_sizes) * channel_size, output_size)



    def forward(self, x):
        # x size: [batch_size, seq_len]
        # After embedding: [batch_size, seq_len, emb_dim = 300] # 16 * 100 * 300 
        # After transposing it: [batch_size, emb_dim, seq_len]

        embed = self.embedding(x).transpose(1, 2) # Trainable embedding
        embed_frozen = self.frozen_embedding(x).transpose(1, 2) # Frozen embedding

        # embed/embed_frozen shape: [batch_size, embed_dim, seq_len]

        combined_embedding = torch.cat((embed, embed_frozen), dim=1)
        # combined_embedding shape: [batch_size, 2*embed_dim, seq_len]

        convs = [F.relu(conv(combined_embedding)) for conv in self.convs]
        # [num_filters, (batch_size, out_dim_i, seq_len_i)] # output dimensions and seq_lens for each convoluational filter may be different

        maxs = [F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2) for conv_out in convs] # Max pool across time
        # After max pooling: [num_filters, (batch_size, channel_size, 1)]
        # After squeezing: [num_filters, (batch_size, channel_size)]

        flattened_maxs = torch.cat(maxs, dim=1)
        # [batch_size, num_filters * channel_size]


        # logits shape: [batch_size, output_size] ; batch size * num classes
        logits = self.fc(self.dropout(flattened_maxs))

        return logits
        
class W2VSequencer(object):
    def __init__(self, gensim_w2v):
        # Performs the same thing as the previous class
        # Except it uses the vocabulary of words from the word embeddings we use

        self.nlp = spacy.load('en')
        self.w2v = gensim_w2v
        self.w2v.add(['<unk>'], [np.random.uniform(low=-1, high=1.0, size=(300,))])

        self.unk_index = self.w2v.vocab.get('<unk>')
        self.tokenizer = lambda text: [t.text for t in self.nlp(text)]

    def encode(self, text):
        # Input will look like:
        # [<s>, w1, w2, ..., wn, </s>]
        sequence = []
        tokens = self.tokenizer(text)
        for token in tokens:

            index = self.w2v.vocab.get(token, self.unk_index).index
            sequence.append(index)

        return sequence

    def create_padded_tensor(self, sequences):
        # Given a list of sequences, pad all to the same length

        max_seq_len = max(len(sequence) for sequence in sequences)
        tensor = torch.full((len(sequences), max_seq_len), 0, dtype=torch.long)

        for i, sequence in enumerate(sequences):
            for j, token in enumerate(sequence):
                tensor[i][j] = token
        
        return tensor
     
def list_to_label(test_pred, all_relations_l):
    final_list = []
    for item in test_pred:
        final_list.append(all_relations_l[int(item)])
    return final_list
            
def prepare_batch(batch, sequencer):
    # batch: [batch_len, (text, label)]
    texts, labels = zip(*batch)
    text_tensor = sequencer.create_padded_tensor(texts)
    return (text_tensor, torch.stack(labels))

def main():
    input_file = os.path.join('clean_data.csv')
    #input_file = 'more_data.csv'
    df = pd.read_csv(input_file, sep=',', engine='python')
    df_data = df.values.tolist()
    #all_relations_d, all_relations_l = get_all_relations(df_data) 
    #print(all_relations_d)
    #all_features_d, all_features_l = get_all_features(df_data)
    y_dict,y_dict_list = gety_dict(df_data)
    X = df['Text'].tolist()
    all_relations_l = [0, 1]
    y = df['Sentiment'].tolist()
    rel_index = []
    for rel in y:
        if rel == -1:
            rel_index.append(0)
        if rel == 1:
            rel_index.append(1)

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(X, torch.LongTensor(rel_index), test_size=0.2, shuffle=True)
    torch.flatten(valid_labels)
    torch.flatten(train_labels)
    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_vec.fit(train_texts)
    #print(tfidf_vec.vocabulary_)
    #print(len(tfidf_vec.vocabulary_))
    #print(len(all_features_l))
    input_size = len(tfidf_vec.vocabulary_)
    hidden_size = 500 # An arbitrary hyperparameter we define
    output_size = len(y_dict_list)
    LEARNING_RATE = 0.1
    loss_fn = nn.CrossEntropyLoss()
    

    input_transformer = lambda text: torch.FloatTensor(tfidf_vec.transform([text]).todense()).squeeze(0)

    train_tfidf_dataset = BBCNewsDataset(train_texts, train_labels, input_transformer)
    valid_tfidf_dataset = BBCNewsDataset(valid_texts, valid_labels, input_transformer)


    df = load_test_data()
    df_data = df.values.tolist()
    X = df['HEADLINE'].tolist()
    check_test = df['SENTIMENT'].tolist()
    test_tfidf_dataset = BBCNewsDataset(X, X, input_transformer)

    train_tfidf_loader = torch.utils.data.DataLoader(train_tfidf_dataset, batch_size=16)
    valid_tfidf_loader = torch.utils.data.DataLoader(valid_tfidf_dataset, batch_size=16)
    test_tfidf_loader = torch.utils.data.DataLoader(test_tfidf_dataset, batch_size=16)
    for train in train_tfidf_loader:
        x,y=train
        f = open("singleclass.txt", "w")
        f.write(str(x.tolist()))
        f.write(str(y))
        f.close()
        break


    # Model 1: MLP with dropout baseline
    mlp_with_dropout = MultiLayerPerceptron(input_size, hidden_size, output_size, dropout=True, dropout_p=0.25)
    mlp_with_dropout_optimizer = optim.Adagrad(mlp_with_dropout.parameters(), lr=LEARNING_RATE)
    mlp_with_dropout_trainer = MultiClassTrainer(mlp_with_dropout, mlp_with_dropout_optimizer, loss_fn)
    mlp_with_dropout_trainer.run_training(train_tfidf_loader, valid_tfidf_loader, y_dict_list, n_epochs=10)
    test_pred = mlp_with_dropout_trainer.run_testing(test_tfidf_loader)


    # Model 2: CNN
    #word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format("model.txt")
    #sequencer = W2VSequencer(word2vec_weights)
    #sequence_input_transformer = lambda text: sequencer.encode(text)
    #train_sequence_dataset = BBCNewsDataset(train_texts, train_labels, sequence_input_transformer)
    #valid_sequence_dataset = BBCNewsDataset(valid_texts, valid_labels, sequence_input_transformer)
    #test_sequence_dataset = BBCNewsDataset(X, torch.FloatTensor([0]*len(X)), sequence_input_transformer)
    #train_sequence_loader = torch.utils.data.DataLoader(train_sequence_dataset, batch_size=50, collate_fn=lambda batch: prepare_batch(batch, sequencer))
    #valid_sequence_loader = torch.utils.data.DataLoader(valid_sequence_dataset, batch_size=50, collate_fn=lambda batch: prepare_batch(batch, sequencer))
    #test_sequence_loader = torch.utils.data.DataLoader(test_sequence_dataset, batch_size=50, collate_fn=lambda batch: prepare_batch(batch, sequencer))
    #cnn = TextConvolver(input_size, output_size, [3, 4, 5], channel_size=100, dropout=True,dropout_p=0.2, w2v_weights=torch.FloatTensor(word2vec_weights.vectors))
    #cnn_optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
    #cnn_trainer = MultiClassTrainer(cnn, cnn_optimizer, loss_fn)
    #cnn_trainer.run_training(train_sequence_loader, valid_sequence_loader, y_dict_list, n_epochs=1)
    #test_pred = cnn_trainer.run_testing(test_sequence_loader)


    final1 = []
    pos = 0
    neg = 0
    for x in test_pred:
        if x == 1:
            final1.append(1)
            pos +=1
        else:
            neg +=1
            final1.append(-1)

    print('pred pos: ',str(pos),'pred neg: ', str(neg))

    test_pos = 0
    test_neg = 0
    correct = 0
    total = len(final1)
    for i in range(len(final1)):
        if final1[i] == check_test[i]:
            correct += 1
        if int(check_test[i]==1):
            test_pos += 1
        else:
            test_neg += 1

    print('test pos: ',str(test_pos),'test neg: ', str(test_neg))
    #print('F1 Score: ', round(f1_score(check_test, test_pred, average='micro'), 3))
    print('Acc:', correct/total)



if __name__ == '__main__':
    main()
