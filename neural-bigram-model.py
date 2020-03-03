import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os.path

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

class Dictionary:
    def __init__(self):
        self.word2index = {'START': 0, 'END': 1}
        self.index2Word = ['START', 'END']
        self.bigrams = []
        self.lastWord = None
        self.wordCount = 2


    def add_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = len(self.word2index.keys())
            self.index2Word.append(word)
            self.wordCount += 1

        if self.lastWord:
            self.bigrams.append((self.lastWord, word))

        self.lastWord = word

    def get_one_hot(self, word):
        word = word.lower()
        oneHot = torch.zeros(self.wordCount)
        oneHot[self.word2index[word]] = 1

        return oneHot

    def word_from_one_hot(self, one_hot):
        # Converts a one hot representation to a word in the dictionnary
        values, indices = one_hot.topk(5)
        return [self.index2Word[index] for index in indices]

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word.lower())

class Net(nn.Module):

    def __init__(self, dic):
        super(Net, self).__init__()

        self.dic = dic

        self.linear1 = nn.Linear(dic.wordCount, 1000)
        self.linear2 = nn.Linear(1000, 500)
        # self.linear3 = nn.Linear(500, 750)
        self.output = nn.Linear(1000, dic.wordCount)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = (self.output(x) / 2.0) + 0.5
        return x


dic = Dictionary()
for sent in brown.sents():
    dic.add_sentence(sent)

net = Net(dic).to(device)

# Load model from file is exists
modelLoaded = False
if os.path.isfile("netral-bigram-model-saved.pth"):
    net.load_state_dict(torch.load("netral-bigram-model-saved.pth"))
    modelLoaded = True

if not modelLoaded:
    # Train the model

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.MSELoss()

    print(dic.wordCount)
    print(dic.word2index)

    # Train on every word
    bigrams = dic.bigrams
    nbTrain = 20000#len(bigrams)

    for i in range(nbTrain):
        bigram = bigrams[i]
        X = dic.get_one_hot(bigram[0]).to(device)
        Y = dic.get_one_hot(bigram[1]).to(device)

        prediction = net(X)

        loss = loss_func(prediction, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 2000 == 0:
            print("{} / {}".format(i, nbTrain))
            print("Loss: %.4f" % loss)

    # Save the trained model
    torch.save(net.state_dict(), "netral-bigram-model-saved.pth")

# Test the model
while True:
    print("Enter a word for suggestion:")
    word = input()

    goIn = dic.get_one_hot(word).to(device)
    out = net(goIn)
    print("Suggestions: {}".format(dic.word_from_one_hot(out)))








