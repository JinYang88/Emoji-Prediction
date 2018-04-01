# coding: utf-8
import pandas as pd
import numpy as np
import re
import logging
import torch
from torchtext import data
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import io
import time
import sys
import model
import datahelper
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

torch.manual_seed(42)

device = 0 # 0 for gpu, -1 for cpu
test_mode = 0  # 0 for train+test 1 for test


bidirectional = True
batch_size = 64
epochs = 1
print_every = 1000

p_dropout = 0.5
embedding_dim = 100
hidden_dim = 100
emoji_num = 20
out_dim = emoji_num



print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=datahelper.normalizeString)
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True)
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)

train = data.TabularDataset(
        path='../data/tweet/multi/top{}/train.csv'.format(emoji_num), format='csv',
        fields=[('Id', ID), ('Text', TEXT), ('Label', LABEL)], skip_header=True)
valid = data.TabularDataset(
        path='../data/tweet/multi/top{}/valid.csv'.format(emoji_num), format='csv',
        fields=[('Id', ID), ('Text', TEXT), ('Label', LABEL)], skip_header=True)
test = data.TabularDataset(
        path='../data/tweet/multi/top{}/test.csv'.format(emoji_num), format='csv',
        fields=[('Id', ID), ('Text', TEXT), ('Label', LABEL)], skip_header=True)


TEXT.build_vocab(train,valid,test, min_freq=3)
print('Building vocabulary Finished.')


train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text), device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=batch_size, device=device, shuffle=False, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=batch_size, device=device, shuffle=False, repeat=False)


train_dl = datahelper.BatchWrapper(train_iter, ["Text", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Text", "Label"])
print('Reading data done.')



class BLSTM(torch.nn.Module) :
    def __init__(self,vocab_size, embedding_dim, hidden_dim, out_dim, batch_size, p_dropout, bidirectional) :
        super(BLSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=bidirectional, dropout=p_dropout)
        self.linearOut = nn.Linear(hidden_dim, out_dim)
    def forward(self,inputs, hidden_state) :
        x = self.embeddings(inputs)
        lstm_out,(lstm_h, lstm_c) = self.lstm(x, hidden_state)
        x = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        x = self.linearOut(x)
        x = F.log_softmax(x, dim=1)

        return x

    def init_hidden(self, batch_size, device) :
        layer_num = 2 if self.bidirectional else 1
        if device == -1:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)))  
        else:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)).cuda(),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)).cuda())  


def predict_on(model, data_dl, loss_func, device, model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    model = model.eval()
    res_list = []
    label_list = []
    loss = 0
    

    for text, labels in data_dl:
        hidden_state = MODEL.init_hidden(text.size()[0], device)
        y_pred = model(text, hidden_state)
        loss += loss_func(y_pred, labels)
        y_pred = y_pred.data.max(1)[1].cpu().numpy()
        res_list.extend(y_pred)
        label_list.extend(labels.data.cpu().numpy())

    acc = accuracy_score(res_list, label_list)
    Precision = precision_score(res_list, label_list, average="macro")
    Recall = recall_score(res_list, label_list, average="macro")
    F1_macro = f1_score(res_list, label_list, average="macro")
    F1_micro = f1_score(res_list, label_list, average="micro")

    if model_state_path:
        with open('BLSTM-Result.txt', 'w') as fw:
            for line in res_list:
                fw.write(str(line) + '\n')
    return loss, (acc, Precision, Recall, F1_macro, F1_micro)


print('Initialing model..')
MODEL = BLSTM(len(TEXT.vocab), embedding_dim, hidden_dim, out_dim, batch_size, p_dropout, bidirectional)
if device == 0:
    MODEL.cuda()


# Train
if not test_mode:
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        train_iter.init_epoch()
        batch_count = 0
        for text, label in train_dl:
            MODEL = MODEL.train(True)
            hidden_state = MODEL.init_hidden(text.size()[0], device)
            y_pred = MODEL(text, hidden_state)
            loss = loss_function(y_pred, label)
            MODEL.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % print_every == 0:
                loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, valid_dl, loss_function, device)
                batch_end = time.time()
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. F1_macro is {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), F1_macro, float(loss)))
        torch.save(MODEL.state_dict(), '../model_save/BLSTM{}.pth'.format(i+1))           


# Test
loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, test_dl, nn.NLLLoss(), device, '../model_save/BLSTM{}.pth'.format(epochs))

print("=================")
print("Evaluation results on test dataset:")
print("Loss: {}.".format(float(loss)))
print("Accuracy: {}.".format(acc))
print("Precision: {}.".format(Precision))
print("Recall: {}.".format(Recall))
print("F1_micro: {}.".format(F1_micro))
print("\n")
print("F1_macro: {}.".format(F1_macro))
print("=================")



