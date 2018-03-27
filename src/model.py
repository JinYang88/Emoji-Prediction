import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class lstm(torch.nn.Module) :
    def __init__(self,vocab_size, embedding_dim, hidden_dim, out_dim, batch_size) :
        super(lstm_model,self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linearOut = nn.Linear(hidden_dim, out_dim)
    def forward(self,inputs) :
        x = self.embeddings(inputs)
        lstm_out,lstm_h = self.lstm(x, None)
        x = lstm_out[:, -1, :]
        x = self.linearOut(x)
        x = F.log_softmax(x, dim=1)

        return x
    def init_hidden(self) :
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))  

class bi_lstm(torch.nn.Module) :
    def __init__(self,vocab_size, embedding_dim, hidden_dim, out_dim, batch_size) :
        super(bi_lstm,self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.8)
        self.linearOut = nn.Linear(hidden_dim, out_dim)
    def forward(self,inputs) :
        x = self.embeddings(inputs)
        lstm_out,(lstm_h, lstm_c) = self.lstm(x, None)
        x = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        x = self.linearOut(x)
        x = F.log_softmax(x, dim=1)

        return x
    def init_hidden(self) :
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))