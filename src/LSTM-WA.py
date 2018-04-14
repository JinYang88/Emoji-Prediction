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

test_mode = 0  # 0 for train+test 1 for test
device = 0 # 0 for gpu, -1 for cpu

bidirectional = True
emoji_num = 5
embedding_dim = 300
hidden_dim = 600

batch_size = 32
epochs = 4
print_every = 500


print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=datahelper.normalizeString)
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
EMOJI = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, preprocessing=normalize_pipeline)
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

TEXT.build_vocab(train,valid,test, min_freq=5)
print('Building vocabulary Finished.')

train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text), device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=batch_size, device=device, shuffle=False, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=batch_size, device=device, shuffle=False, repeat=False)


train_dl = datahelper.BatchWrapper(train_iter, ["Text", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Text", "Label"])
print('Reading data done.')


def predict_on(model, data_dl, loss_func, device ,model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    model.eval()
    res_list = []
    label_list = []
    loss = 0

    for text, label in data_dl:
        hidden_state = MODEL.init_hidden(text.size()[0], device)
        y_pred = MODEL(text, hidden_state)
        
        loss += loss_func(y_pred, label).data.cpu()
        y_pred = y_pred.data.max(1)[1].cpu().numpy()
        res_list.extend(y_pred)
        label_list.extend(label.data.cpu().numpy())

    acc = accuracy_score(res_list, label_list)
    Precision = precision_score(res_list, label_list, average="macro")
    Recall = recall_score(res_list, label_list, average="macro")
    F1_macro = f1_score(res_list, label_list, average="macro")
    F1_micro = f1_score(res_list, label_list, average="micro")

    if model_state_path:
        with open('LSTM_A-Result.txt', 'w') as fw:
            for line in res_list:
                fw.write(str(line) + '\n')
    return loss, (acc, Precision, Recall, F1_macro, F1_micro)



class LSTM_WA(torch.nn.Module) :
    def __init__(self, vocab_size, emoji_num, embedding_dim, hidden_dim, batch_size, device, bidirectional):
        super(LSTM_WA,self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.emoji_num = emoji_num
        self.batch_size = batch_size
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emoji_matrix = torch.nn.Parameter(torch.rand(emoji_num, embedding_dim))
        self.cosine_similarity = F.cosine_similarity
        self.lstm = nn.LSTM(embedding_dim * emoji_num, hidden_dim // 2 if self.bidirectional else hidden_dim, batch_first=True, bidirectional=self.bidirectional)

        # self.linear1 = nn.Linear(hidden_dim, 200)
        # self.dropout1 = nn.Dropout(p=0.1)
        # self.linear2 = nn.Linear(200, emoji_num)

        self.linear1 = nn.Linear(hidden_dim, 200)
        self.dropout1 = nn.Dropout(p=0.5)
        # self.batchnorm1 = nn.BatchNorm1d(200)
        self.linear2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(p=0.5)
        # self.batchnorm2 = nn.BatchNorm1d(200)
        self.linear3 = nn.Linear(200, 200)
        self.dropout3 = nn.Dropout(p=0.5)
        # self.batchnorm3 = nn.BatchNorm1d(200)
        self.linear4 = nn.Linear(200, 200)
        self.dropout4 = nn.Dropout(p=0.5)
        # self.batchnorm4 = nn.BatchNorm1d(200)
        self.linear5 = nn.Linear(200, emoji_num)
        
    def forward(self, text, hidden_init) :
        word_embedding = self.word_embedding(text)
        seq_embedding = self.attention(word_embedding, self.emoji_matrix)
        lstm_out,(lstm_h, lstm_c) = self.lstm(seq_embedding, hidden_init)
 
        if not self.bidirectional:
            linear_in = lstm_h.squeeze(0)
        else:
            linear_in = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        
        # linearo = self.linearOut(linear_in)

        merged = self.linear1(linear_in)
        merged = F.tanh(merged)
        merged = self.dropout1(merged)

        merged = self.linear2(merged)
        merged = F.tanh(merged)
        merged = self.dropout2(merged)

        # merged = self.linear3(merged)
        # merged = F.relu(merged)
        # merged = self.dropout3(merged)

        # merged = self.linear4(merged)
        # merged = F.relu(merged)
        # merged = self.dropout4(merged)

        merged = self.linear5(merged)

        return F.log_softmax(merged, dim=1)
        
        
    def attention(self, lstm_out, emoji_matrix):
        seq_embeddings = []
        for emoji_idx in range(self.emoji_num):
            similarities = self.cosine_similarity(lstm_out, emoji_matrix[emoji_idx].unsqueeze(0), dim=-1)
            simi_weights = F.softmax(similarities, dim=1).view(lstm_out.size()[0], -1, 1)
            seq_embedding = simi_weights * lstm_out
            seq_embeddings.append(seq_embedding)
        seq_embeddings = torch.cat(seq_embeddings, dim=2)
        return seq_embeddings


    def init_hidden(self, batch_size, device) :
        layer_num = 2 if self.bidirectional else 1
        if device == -1:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num), requires_grad=False),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)))  
        else:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda(), requires_grad=False),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda(), requires_grad=False))  



print('Initialing model..')
MODEL = LSTM_WA(len(TEXT.vocab), emoji_num, embedding_dim, hidden_dim, batch_size, device, bidirectional)
if device == 0:
    MODEL.cuda()
    
# print(MODEL.state_dict())

# sys.exit()
best_state = None
max_metric = 0

# Train
if not test_mode:
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        train_iter.init_epoch()
        batch_count = 0
        for text, label in train_dl:
            MODEL.train(True)
            hidden_state = MODEL.init_hidden(text.size()[0], device)
            y_pred = MODEL(text, hidden_state)
            loss = loss_func(y_pred, label)
            MODEL.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % print_every == 0:
                loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, valid_dl, loss_func, device)
                batch_end = time.time()
                if F1_micro > max_metric:
                    best_state = MODEL.state_dict()
                    max_metric = F1_micro
                    print("Saving model..")
                    torch.save(best_state, '../model_save/LSTM_WA.pth')           
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. F1_micro is {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), F1_micro, float(loss)))
        


loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, test_dl, nn.NLLLoss(), device, '../model_save/LSTM_WA.pth')

print("=================")
print("Evaluation results on test dataset:")
print("Loss: {}.".format(float(loss)))
print("Accuracy: {}.".format(acc))
print("Precision: {}.".format(Precision))
print("Recall: {}.".format(Recall))
print("F1_micro: {}.".format(F1_micro))
print("F1_macro: {}.".format(F1_macro))
print("=================")                        