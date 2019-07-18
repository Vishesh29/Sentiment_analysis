from __future__ import print_function
import re
import csv
import pandas as pd
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D
from keras.layers import LSTM
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")
frames = [train, test]
df = pd.concat(frames)
# Shuffling the data
df = df.sample(frac=1)

# printing the first few lines of the dataframe
df.head()

# Visualizing the characteristics of classes in the dataset
class_stat={}
class_stat['0'] = df[df['Sentiment']==0].shape[0]
class_stat['1'] = df[df['Sentiment']==1].shape[0]
class_stat['2'] = df[df['Sentiment']==2].shape[0]
class_stat['3'] = df[df['Sentiment']==3].shape[0]
class_stat['4'] = df[df['Sentiment']==4].shape[0]
plt.bar(range(len(class_stat)), list(class_stat.values()), align='center')
plt.xticks(range(len(class_stat)), list(class_stat.keys()))
plt.xlabel("Class Labels")
plt.ylabel("Count of sentences")
plt.title("Visualization of all the classes in the dataset")
plt.show()

print("Printing the shape of dataframe:",df.shape)
phrase_list = list(df['Phrase'])
print("Printing the first two sentences:", phrase_list[:2])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

phrase_list = [clean_text(text) for text in phrase_list]

# Creating a dictionary to hold all the unique token indexes
vocab_dict = {} # consist the vocabulary for the corpus
token = 0 # For token index
for sent in phrase_list:    
# Getting words from a sentence
    words = nltk.word_tokenize(sent)
# Inserting start of sentence and end of sentence
    words.insert(0,"<pad>")
    words.insert(1,"<sos>")
    words.insert(len(words),"<eos>")
    # Adding tokens to dictionary as per their position of occurence
    for word in words:
        if word.lower() not in vocab_dict.keys():
            vocab_dict[word.lower()] = token
            token += 1
        else:
            pass

print("size of the vocabulaury :",len(vocab_dict))

print("Printing the shape of training dataframe:",train.shape)
train_text = list(train['Phrase'])
print("Printing the first two sentences:", train_text[:2])

# Cleaning the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

train_text = [clean_text(text) for text in train_text]

# Using the dictionary to vectorize the sentences.
tokenized_sent = []
for sent in train_text:
    # Tokenizes the sentences into list of words
    sent = nltk.word_tokenize(sent)
    # Inserting start of sentence and end of sentence
    sent.insert(0,"<sos>")
    sent = [vocab_dict[token.lower()] for token in sent]
    tokenized_sent.append(sent)

# Vectorizing sentences in dataframe
train['Phrase'] = tokenized_sent
train = train[np.isfinite(train['Sentiment'])]
train.head()

print("Printing the shape of training dataframe:",test.shape)
test_text = list(test['Phrase'])
print("Printing the first two sentences:", test_text[:2])

# Cleaning the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

test_text = [clean_text(text) for text in test_text]

# Using the dictionary to vectorize the sentences.

tokenized_sent = []
for sent in test_text:
    # Tokenizes the sentences into list of words
    sent = nltk.word_tokenize(sent)
    # Inserting start of sentence and end of sentence
    sent.insert(0,"<sos>")
    sent.insert(len(sent),"<eos>")
    sent = [vocab_dict[token.lower()] for token in sent]
    tokenized_sent.append(sent)

# Vectorizing sentences in dataframe
test['Phrase'] = tokenized_sent
#test = test[np.isfinite(test['Sentiment'])]
test.head()

X_train = np.array(train['Phrase'])
y_train = np.array(train['Sentiment'])
X_test = np.array(test['Phrase'])
length_X_train = [len(sent) for sent in X_train]
length_X_test = [len(sent) for sent in X_test]
max_length_X_train = max(length_X_train)
max_length_X_test = max(length_X_test)
max_length = max(max_length_X_train,max_length_X_test)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(X_train, maxlen=max_length)
x_test = sequence.pad_sequences(X_test, maxlen=max_length)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape',y_train.shape)
      
#Splitting data into training and validation set 
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
# Model configurations
max_features = len(vocab_dict)
batch_size = 32
nhid = 64
epochs = 10
print('Building model...')
model = Sequential()
model.add(Embedding(max_features, nhid))
model.add(LSTM(nhid, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(32, activation='softmax'))
#model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
#model.add(Dense(16, activation='softmax'))
#model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(list(train['Sentiment'].unique())), activation='softmax'))
# try using different optimizers and different optimizer configs
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

score = model.evaluate(X_test, y_test,batch_size=batch_size, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# Saving the model
model.save('lstm_model.h5')  # creates a HDF5 file 'my_model.h5'
# Loading the model
model = load_model('lstm_model.h5')
phrase_id = list(test['PhraseId'])
value_dict = {}
for index, test_sequences in enumerate(x_test):
    test_sequences = np.expand_dims(test_sequences, axis=0)
    value_dict[phrase_id[index]] = int(np.argmax(model.predict(test_sequences)[0]))

df = pd.DataFrame.from_dict(value_dict,orient='index')
df['PhraseId'] = df.index
df.columns = ['Sentiment', 'PhraseId']
df.to_csv("output_sentiment.tsv", sep='\t')
