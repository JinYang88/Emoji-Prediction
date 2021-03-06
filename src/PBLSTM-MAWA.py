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

test_mode = 0  # 0 for train+test 1 for test
device = 0 # 0 for gpu, -1 for cpu

bidirectional = False
emoji_num = 5
embedding_dim = 300
hidden_dim = embedding_dim
out_dim = 1
p_dropout = 0.1

batch_size = 64
epochs = 4
print_every = 1

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


TEXT.build_vocab(train, valid, test, min_freq=5)
print('Building vocabulary Finished.')


train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text), device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=batch_size, device=device, shuffle=False, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=batch_size, device=device, shuffle=False, repeat=False)

train_dl = datahelper.BatchWrapper(train_iter, ["Text", "Emoji", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Id", "Text", "Emoji", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Id", "Text", "Emoji", "Label"])
print('Reading data done.')


emoji_matrix = datahelper.wordlist_to_matrix("../data/embedding/top5embedding.txt", ["<{}>".format(i) for i in range(emoji_num)], device, embedding_dim)
word_matrix = datahelper.wordlist_to_matrix("../data/embedding/top5embedding.txt", TEXT.vocab.itos, device, embedding_dim)


# data_dl: id, text, emoji, label
def predict_on(model, data_dl, loss_func, device ,model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    model.eval()
    result_list = []  # id, emoji, similarity, label
    id_list = []
    emoji_list = []
    similarity_list = []
    labels_list = []
    loss = 0
    for ids, text, emoji, label in data_dl:
        hidden_state = MODEL.init_hidden(text.size()[0], device)
        similarity = MODEL(text, emoji.view(-1,1), hidden_state)
        loss += loss_func(similarity, label.view(-1,1).float()).data.cpu()
        id_list.extend(ids.data.cpu().numpy())
        emoji_list.extend(emoji.data.cpu().numpy())
        similarity_list.extend(similarity.data.cpu().numpy())
        labels_list.extend(label.data.cpu().numpy())
        
        
    result_df = pd.DataFrame()
    result_df['id'] = id_list
    result_df['emoji'] = emoji_list
    result_df['similarity'] = similarity_list
    result_df['label'] = labels_list
    answer_df = result_df.loc[result_df.groupby("id")['similarity'].idxmax().values][['id','emoji']].rename(columns={"emoji":"prediction"})
    ground_truth_df = result_df[result_df['label']==1].rename(columns={"emoji":"groundtruth"})
    final_df = answer_df.merge(ground_truth_df, on="id")
    acc = accuracy_score(final_df['prediction'], final_df['groundtruth'])
    Precision = precision_score(final_df['prediction'], final_df['groundtruth'], average="macro")
    Recall = recall_score(final_df['prediction'], final_df['groundtruth'], average="macro")
    F1_macro = f1_score(final_df['prediction'], final_df['groundtruth'], average="macro")
    F1_micro = f1_score(final_df['prediction'], final_df['groundtruth'], average="micro")
    return loss, (acc, Precision, Recall, F1_macro, F1_micro)

class PBLSTM_MAWA(torch.nn.Module) :
    def __init__(self, vocab_size, emoji_num, embedding_dim, hidden_dim, batch_size, bidirectional, word_mat, emoji_mat):
        super(PBLSTM_MAWA,self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(word_mat)
        self.word_embedding.weight.requires_grad = False
        
        self.emoji_embedding = nn.Embedding(emoji_num, embedding_dim)
        self.emoji_embedding.weight.data.copy_(emoji_mat)
        self.emoji_embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim //2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.cosine_similarity = F.cosine_similarity
        
        self.linearOut = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, text, emoji, hidden_init) :

        word_embedding = self.word_embedding(text)
        emoji_embedding = self.emoji_embedding(emoji)  # batchsize, 1, embedding_dim

#         print(word_embedding.size())
#         print(emoji_embedding.size())
        
        similarities = self.cosine_similarity(emoji_embedding, word_embedding, dim=2)
#         print(similarities)
        simi_weights = F.softmax(similarities, dim=1).view(-1, word_embedding.size()[1], 1)
        weighted_embedding = simi_weights * word_embedding
        #lstm_out batch_size, sequence len, embedding_dim
        lstm_out,(lstm_h, lstm_c) = self.lstm(weighted_embedding, hidden_init)
        
        if self.bidirectional:
            seq_embedding = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        else:
            seq_embedding = lstm_h.view(-1,1,self.hidden_dim)
            
            
        linearo = self.linearOut(seq_embedding)
        similarity = self.cosine_similarity(linearo.squeeze(1), emoji_embedding.squeeze(1))

        return similarity
        

    def init_hidden(self, batch_size, device) :
        layer_num = 2 if self.bidirectional else 1
        if device == -1:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num), requires_grad=False),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num),requires_grad=False))  
        else:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda(),requires_grad=False),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda(), requires_grad=False))  




print('Initialing model..')
torch.backends.cudnn.benchmark = True 
MODEL = PBLSTM_MAWA(len(TEXT.vocab),emoji_num, embedding_dim,
                   hidden_dim, batch_size, bidirectional, word_matrix, emoji_matrix)
best_state = None
max_metric = 0

if device == 0:
    MODEL.cuda()

# Train
if not test_mode:
    loss_func = nn.MSELoss()
    parameters = list(filter(lambda p: p.requires_grad, MODEL.parameters()))
    optimizer = optim.Adam(parameters, lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        train_iter.init_epoch()
        batch_count = 0
        for text, emoji, label in train_dl:
            MODEL.train()
            hidden_state = MODEL.init_hidden(text.size()[0], device)
            similarity = MODEL(text, emoji.view(-1,1), hidden_state)
            loss = loss_func(similarity.view(-1,1), label.view(-1,1).float())
            loss.backward()
            optimizer.step()
            MODEL.zero_grad()
            batch_count += 1
            # print("finish a batch")
            if batch_count % print_every == 0:
                loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, valid_dl, loss_func, device)
                batch_end = time.time()
                if F1_micro > max_metric:
                    best_state = MODEL.state_dict()
                    max_metric = F1_micro
                    print("Saving model..")
                    torch.save(best_state, '../model_save/PBLSTM_MAWA.pth')           
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. F1_micro is {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), F1_micro, float(loss)))
        


loss, (acc, Precision, Recall, F1_macro, F1_micro) = predict_on(MODEL, test_dl,nn.MSELoss(), device, '../model_save/PBLSTM_MAWA.pth')

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