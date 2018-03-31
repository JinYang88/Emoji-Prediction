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


test_mode = 1  # 0 for train+test 1 for test
device = -1 # 0 for gpu, -1 for cpu

bidirectional = True
emoji_num = 20
embedding_dim = 100
hidden_dim = embedding_dim
out_dim = 1
p_dropout = 0.1

batch_size = 32
epochs = 4
print_every = 50


print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=datahelper.normalizeString)
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
EMOJI = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, preprocessing=normalize_pipeline)
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)


train = data.TabularDataset(
        path='../data/tweet/binary/top{}/train.csv'.format(emoji_num), format='csv',
        fields=[('Id', ID), ('Text', TEXT),('Emoji', EMOJI), ('Label', LABEL)], skip_header=True)
valid = data.TabularDataset(
        path='../data/tweet/binary/top{}/valid.csv'.format(emoji_num), format='csv',
        fields=[('Id', ID), ('Text', TEXT),('Emoji', EMOJI), ('Label', LABEL)], skip_header=True)
test = data.TabularDataset(
        path='../data/tweet/binary/top{}/test.csv'.format(emoji_num), format='csv',
        fields=[('Id', ID), ('Text', TEXT),('Emoji', EMOJI), ('Label', LABEL)], skip_header=True)

TEXT.build_vocab(train,valid,test, min_freq=3)
print('Building vocabulary Finished.')


train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text), device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=len(valid.examples), device=device, shuffle=False, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=len(test.examples), device=device, shuffle=False, repeat=False)

train_dl = datahelper.BatchWrapper(train_iter, ["Text", "Emoji", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Id", "Text", "Emoji", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Id", "Text", "Emoji", "Label"])
print('Reading data done.')



class lstm_match(torch.nn.Module) :
    def __init__(self, vocab_size, emoji_num, embedding_dim, hidden_dim, batch_size, bidirectional, dropout):
        super(lstm_match,self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emoji_embedding = nn.Embedding(emoji_num, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        
    def forward(self,text, emoji, hidden_init) :
        word_embedding = self.word_embedding(text)
        emoji_embedding = self.emoji_embedding(emoji)
        lstm_out,(lstm_h, lstm_c) = self.lstm(word_embedding, hidden_init)
        
#         print(lstm_h)
        
        if self.bidirectional:
            seq_embedding = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        else:
            seq_embedding = lstm_h.view(self.batch_size,1,-1)
            
#         print(seq_embedding.size(), emoji_embedding.size())

        return seq_embedding, emoji_embedding
        
    
    def init_hidden(self, batch_size) :
        return (Variable(torch.randn(batch_size, batch_size, self.hidden_dim)),Variable(torch.randn(1, batch_size, self.hidden_dim)))  


    
# data_dl: id, text, emoji, label
def predict_on(model, data_dl, model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    loss_func = nn.CosineEmbeddingLoss()
    model = MODEL.eval()
    result_list = []  # id, emoji, similarity, label
    for ids, text, emoji, label in data_dl:
        seq_embedding, emoji_embedding = MODEL(text, emoji.view(-1,1), None)
        p_pred = F.cosine_similarity(seq_embedding.squeeze(1), emoji_embedding.squeeze(1))
        loss = loss_func(seq_embedding.squeeze(1), emoji_embedding.squeeze(1), label.view(-1,1))
        result_nparr = ids.data.cpu().numpy(), emoji.data.cpu().numpy(), p_pred.data.cpu().numpy(), label.data.cpu().numpy()
    result_df = pd.DataFrame()
    result_df['id'] = result_nparr[0]
    result_df['emoji'] = result_nparr[1]
    result_df['similarity'] = result_nparr[2]
    result_df['label'] = result_nparr[3]
    answer_df = result_df.loc[result_df.groupby("id")['similarity'].idxmax().values][['id','emoji']].rename(columns={"emoji":"prediction"})
    ground_truth_df = result_df[result_df['label']==1].rename(columns={"emoji":"groundtruth"})
    final_df = answer_df.merge(ground_truth_df, on="id")
    final_df.to_csv("lstm_match_results.csv", index=False)
    acc = accuracy_score(final_df['prediction'], final_df['groundtruth'])
    Precision = precision_score(final_df['prediction'], final_df['groundtruth'], average="macro")
    Recall = recall_score(final_df['prediction'], final_df['groundtruth'], average="macro")
    F1_macro = f1_score(final_df['prediction'], final_df['groundtruth'], average="macro")
    F1_micro = f1_score(final_df['prediction'], final_df['groundtruth'], average="micro")
    return loss, (acc, Precision, Recall, F1_macro, F1_micro)


print('Initialing model..')
MODEL = lstm_match(len(TEXT.vocab),emoji_num, embedding_dim,
                   hidden_dim, batch_size, bidirectional, p_dropout)
if device == 0:
    MODEL.cuda()

# Train
if not test_mode:
    loss_func = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        train_iter.init_epoch()
        batch_count = 0
        for text, emoji, label in train_dl:
            seq_embedding, emoji_embedding = MODEL(text, emoji.view(-1,1), None)
            loss = loss_func(seq_embedding.squeeze(1), emoji_embedding.squeeze(1), label.view(-1,1))
            loss.backward()
            optimizer.step()
            MODEL.zero_grad()
            batch_count += 1
            if batch_count % print_every == 0:
                loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, valid_dl)
                batch_end = time.time()
                MODEL = MODEL.train(True)
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. F1_macro is {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), F1_macro, float(loss)))
            torch.save(MODEL.state_dict(), 'lstm_match_model{}.pth'.format(i+1))           
            print("Saving model..")



loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, test_dl, 'lstm_match_model{}.pth'.format(epochs))

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
