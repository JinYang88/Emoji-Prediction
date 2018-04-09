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

emoji_num = 5
embedding_dim = 300

batch_size = 3
epochs = 4
print_every = 100


print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=datahelper.normalizeString)
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
EMOJI = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True,)
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


word_matrix = datahelper.wordlist_to_matrix("../data/embedding/top5embedding.txt", TEXT.vocab.itos, device, embedding_dim)

def predict_on(model, data_dl, loss_func, device ,model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    model = model.eval()
    res_list = []
    label_list = []
    loss = 0

    for text, label in data_dl:
        y_pred = MODEL(text)
        
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
        with open('SGVA-Result.txt', 'w') as fw:
            for line in res_list:
                fw.write(str(line) + '\n')
    return loss, (acc, Precision, Recall, F1_macro, F1_micro)


class SGVA(torch.nn.Module) :
    def __init__(self, vocab_size, emoji_num, embedding_dim, wordvec_matrix):
        super(SGVA,self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(wordvec_matrix)
        self.word_embedding.weight.requires_grad = False
        
        self.linearOut = nn.Linear(embedding_dim, emoji_num)
        
        
    def forward(self, text) :
        word_embeddings = self.word_embedding(text)
        sequence_embeddings = torch.mean(word_embeddings, dim=1)
        linearo = self.linearOut(sequence_embeddings)
        return F.log_softmax(linearo, dim=1)
    

print('Initialing model..')
MODEL = SGVA(len(TEXT.vocab), emoji_num, embedding_dim, word_matrix)
if device == 0:
    MODEL.cuda()
    
# print(MODEL.state_dict())

# sys.exit()
best_state = None
max_metric = 0

# Train
if not test_mode:
    loss_func = nn.NLLLoss()
    parameters = list(filter(lambda p: p.requires_grad, MODEL.parameters()))
    optimizer = optim.Adam(parameters, lr=1e-2)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        train_iter.init_epoch()
        batch_count = 0
        for text, label in train_dl:
            MODEL = MODEL.train(True)
            y_pred = MODEL(text)
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
                    torch.save(best_state, '../model_save/SGVA.pth')           
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. F1_micro is {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), F1_micro, float(loss)))
        


loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, test_dl,nn.NLLLoss(), device, '../model_save/SGVA.pth')

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