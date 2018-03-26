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
from sklearn.metrics import accuracy_score



device = -1 # 0 for gpu, -1 for cpu
batch_size = 64
test_mode = 0  # 0 for train+test 1 for test
embedding_dim = 100
hidden_dim = 64
out_dim = 20
epochs = 1
print_every = 1


print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=datahelper.normalizeString)
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, preprocessing=normalize_pipeline)
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)

train = data.TabularDataset(
        path='../data/tweet/train.csv', format='csv',
        fields=[('Id', ID), ('Text', TEXT), ('Label', LABEL)], skip_header=True)
valid = data.TabularDataset(
        path='../data/tweet/valid.csv', format='csv',
        fields=[('Id', ID), ('Text', TEXT), ('Label', LABEL)], skip_header=True)
test = data.TabularDataset(
        path='../data/tweet/test.csv', format='csv',
        fields=[('Id', ID), ('Text', TEXT), ('Label', LABEL)], skip_header=True)


TEXT.build_vocab(train,valid,test)
print('Building vocabulary Finished.')


train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text), device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=len(valid.examples), device=device, shuffle=False, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=len(test.examples), device=device, shuffle=False, repeat=False)


train_dl = datahelper.BatchWrapper(train_iter, "Text", ["Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, "Text", ["Label"])
test_dl = datahelper.BatchWrapper(test_iter, "Text", ["Label"])
print('Reading data done.')


print('Initialing model..')
MODEL = model.bi_lstm(len(TEXT.vocab), embedding_dim, hidden_dim, out_dim, batch_size)
if device == 0:
    MODEL.cuda()

# Train
if not test_mode:
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    for i in range(epochs) :
        avg_loss = 0.0
        train_iter.init_epoch()
        batch_count = 0
        batch_start = time.time()
        for batch, label in train_dl:
            y_pred = MODEL(batch)
            loss = loss_function(y_pred, label)
            MODEL.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % print_every == 0:
                for batch, labels in valid_dl:
                    MODEL = MODEL.train(False)
                    y_pred = MODEL(batch)
                    y_pred = y_pred.data.max(1)[1].cpu().numpy()
                    acc = accuracy_score(y_pred, labels.cpu().data.numpy())
                batch_end = time.time()
                MODEL = MODEL.train(True)
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. Valid_acc is {}, loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), acc ,float(loss)))
        torch.save(MODEL.state_dict(), 'model' + str(i+1)+'.pth')           


# Test
print('Start predicting...')
MODEL.load_state_dict(torch.load('model{}.pth'.format(epochs)))
MODEL = MODEL.eval()

final_res = []
final_labels = []
for batch, label in test_dl:
    hidden = MODEL.init_hidden()
    y_pred = MODEL(batch)
    pred_res = y_pred.data.max(1)[1].cpu().numpy()
    final_res.extend(pred_res)
    final_labels.extend(list(label.cpu().data))

acc = accuracy_score(final_res, final_labels)
print('Prediction done. Test accuracy is [{}]'.format(acc))







