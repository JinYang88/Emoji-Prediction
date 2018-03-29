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
batch_size = 4
test_mode = 0  # 0 for train+test 1 for test
embedding_dim = 300
hidden_dim = 100
out_dim = 1
epochs = 1
print_every = 10
dtype = torch.FloatTensor if device == -1 else torch.cuda.FloatTensor
perspective_dim = 6


print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=datahelper.normalizeString)
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, preprocessing=normalize_pipeline, use_vocab=True)
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)

train = data.TabularDataset(
        path='../data/quora/train.tsv', format='tsv',
        fields=[('Label', LABEL), ('Text1', TEXT), ('Text2', TEXT), ('Id', ID)], skip_header=True)
valid = data.TabularDataset(
        path='../data/quora/dev.tsv', format='tsv',
        fields=[('Label', LABEL), ('Text1', TEXT), ('Text2', TEXT), ('Id', ID)], skip_header=True)
test = data.TabularDataset(
        path='../data/quora/test.tsv', format='tsv',
        fields=[('Label', LABEL), ('Text1', TEXT), ('Text2', TEXT), ('Id', ID)], skip_header=True)

TEXT.build_vocab(train,test)
text_vocab = TEXT.vocab
word_vec = datahelper.load_glove_as_dict(filepath="../data/quora/wordvec.txt")
print('Building vocabulary Finished.')

word_vec_list = []
for idx, word in enumerate(text_vocab.itos):
    if word in word_vec:
        vector = np.array(word_vec[word], dtype=float).reshape(1,300)
    else:
        vector = np.random.rand(1, 300)
    word_vec_list.append(torch.from_numpy(vector))
wordvec_matrix = torch.cat(word_vec_list)


train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text1) + len(x.Text2), device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=batch_size, device=device, shuffle=False, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=batch_size, device=device, shuffle=False, repeat=False)


train_dl = datahelper.BatchWrapper(train_iter, ["Text1", "Text2", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text1", "Text2", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Text1", "Text2", "Label"])
print('Reading data done.')

train_dl = datahelper.BatchWrapper(train_iter, ["Text1", "Text2", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text1", "Text2", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Text1", "Text2", "Label"])
print('Reading data done.')


print('Initialing model..')
MODEL = model.bi_mpm(len(TEXT.vocab), len(TEXT.vocab), embedding_dim, hidden_dim, out_dim, perspective_dim, wordvec_matrix , batch_size, dtype)
if device == 0:
    MODEL.cuda()

# Train
if not test_mode:
    loss_function = F.binary_cross_entropy
    parameters = list(filter(lambda p: p.requires_grad, MODEL.parameters()))
    optimizer = optim.Adam(parameters, lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        avg_loss = 0.0
        train_iter.init_epoch()
        batch_count = 0
        for text1, text2, label in train_dl:
            y_pred = MODEL(text1,text2)
            loss = loss_function(y_pred, label.float().view(batch_size, 1))
            MODEL.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % print_every == 0:
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. '.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2)))
                # MODEL = MODEL.train(False)
                # final_res = []
                # final_labels = []
                # for text1, text2, label in valid_dl:
                #     y_pred = MODEL(text1,text2)
                #     y_pred = np.array([1 if _ > 0.5 else 0 for _ in y_pred.cpu().data.numpy()])
                #     final_res.extend(y_pred)
                #     final_labels.extend(list(label.cpu().data))
                # acc = accuracy_score(final_res, final_labels)
                # batch_end = time.time()
                # MODEL = MODEL.train(True)
                # print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. Valid_acc is {}, loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), acc ,float(loss)))
        torch.save(MODEL.state_dict(), 'model' + str(i+1)+'.pth')


# Test
print('Start predicting...')
MODEL.load_state_dict(torch.load('model{}.pth'.format(epochs)))
MODEL = MODEL.eval()

final_res = []
final_labels = []
for text1, text2, label in test_dl:
    y_pred = MODEL(text1,text2)
    y_pred = np.array([1 if _ > 0.5 else 0 for _ in y_pred.cpu().data.numpy()])
    final_res.extend(y_pred)
    final_labels.extend(list(label.cpu().data))


Acc = accuracy_score(final_res, final_labels)
Precision = precision_score(final_res, final_labels, average="macro")
Recall = recall_score(final_res, final_labels, average="macro")
F1_macro = f1_score(final_res, final_labels, average="macro")
F1_micro = f1_score(final_res, final_labels, average="micro")
print('Prediction done.')
print('Test accuracy : [{}], Prediction: [{}], Recall: [{}], F1_micro: [{}]'.format(Acc, Precision, Recall, F1_micro))
print('F1_macro: [{}]'.format(F1_macro))

with open('Quora-Prediction-Result.txt', 'w') as fw:
    for line in final_res:
        fw.write(str(line) + '\n')