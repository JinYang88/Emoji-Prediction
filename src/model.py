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




class bi_mpm(torch.nn.Module) :
    def __init__(self, vocab_size, emoji_size, embedding_dim, hidden_dim, out_dim, perspective_dim, wordvec_matrix, batch_size, device) :
        super(bi_mpm,self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.wordvec_matrix = wordvec_matrix
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(self.wordvec_matrix)
        self.word_embedding.weight.requires_grad = False

        self.word_lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.1)
        self.emoji_lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.1)
        self.word_lstm2 = nn.LSTM(perspective_dim * 2, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.1)
        self.emoji_lstm2 = nn.LSTM(perspective_dim * 2, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.1)
        
        if device == 0: 
            self.W1 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
            self.W2 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
            self.W3 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
            self.W4 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
            self.W5 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
            self.W6 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
            self.W7 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
            self.W8 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2).cuda(), requires_grad=True)
        else:
            self.W1 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
            self.W2 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
            self.W3 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
            self.W4 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
            self.W5 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
            self.W6 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
            self.W7 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
            self.W8 = torch.nn.Parameter(torch.randn(perspective_dim, hidden_dim//2), requires_grad=True)
        
        self.linearOut = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, words, emojis) :
        word_embeddings = self.word_embedding(words)
        emoji_embeddings = self.word_embedding(emojis)
#         emoji_embeddings = self.emoji_embedding(emojis)
        
        
        # xxxos contain all hidden outputs, reversed direction should use 0 to index the final output
        # xxxos size is [batch_size X token_counts X hidden_size*num_direction]
        wordos,  (wordo, wordoh) = self.word_lstm(word_embeddings, None)
        emojios, (emojio, emojioh) = self.emoji_lstm(emoji_embeddings, None)

        words_new_seq = self.emoji2word(wordos, emojios)
        emoji_new_seq = self.emoji2word(emojios, wordos)  # emoji side is to take one token, word side take whole sequence
    
        wordos2,  (wordo2, wordoh2) = self.word_lstm2(words_new_seq, None)
        emojios2, (emojio2, emojioh2) = self.emoji_lstm2(emoji_new_seq, None)
        
        words_final_embedding = torch.cat((wordo2[0], wordo2[1]), dim=1)
        emoji_final_embedding = torch.cat((emojio2[0], emojio2[1]), dim=1)
        
        
        pair_embedding = torch.cat([words_final_embedding, emoji_final_embedding], dim=1)
        
#         print(pair_embedding)

        x = self.linearOut(pair_embedding)
        x = F.sigmoid(x)
        return x
    
    def emoji2word(self, emojios, wordos):
        emoji_stamps = []
        for emoji_idx in range(emojios.shape[0]): # get a sample [a sequence]
            for pos in range(emojios.shape[1]): # get a emoji hidden embedding [a token]
                emoji_hidden = emojios[emoji_idx]
                for_emoji_vec = emoji_hidden[pos][0: self.hidden_dim//2]
                bac_emoji_vec = emoji_hidden[emojios.shape[1] - pos - 1][self.hidden_dim//2:]

                # Full Matching:
                final_for_vec = wordos[emoji_idx][-1][0: self.hidden_dim//2]
                final_bac_vec = wordos[emoji_idx][0][self.hidden_dim//2:]
                
                m_full_for = self.fm(for_emoji_vec, final_for_vec, self.W1).view(1,-1)  # vec1
                m_full_bac = self.fm(bac_emoji_vec, final_bac_vec, self.W2).view(1,-1)  # vec2
                
                # Maxpooling Matching:
                m_for_vecs = []
                m_bac_vecs = []
                simi_values_for = [] # for attentive matching
                simi_values_bac = [] # for attentive matching
                for j in range(wordos.shape[1]):
                    for_word_vec = wordos[emoji_idx][j][0: self.hidden_dim//2]
                    bac_word_vec = wordos[emoji_idx][j][self.hidden_dim//2:]
                    m_for_vecs.append(self.fm(for_emoji_vec, for_word_vec, self.W3))
                    m_bac_vecs.append(self.fm(bac_emoji_vec, bac_word_vec, self.W4))
                    simi_values_for.append(self.cosine_sim(for_emoji_vec, for_word_vec))
                    simi_values_bac.append(self.cosine_sim(bac_emoji_vec, bac_word_vec))
                    
                m_for_mat = torch.cat(m_for_vecs)
                m_bac_mat = torch.cat(m_bac_vecs)
                
                
                m_max_for = torch.max(m_for_mat, dim=0)[0].view(1, -1) # vec3
                m_max_bac = torch.max(m_bac_mat, dim=0)[0].view(1, -1) # vec4

                
                simi_values_for = torch.cat(simi_values_for).view(-1, 1)
                simi_values_bac = torch.cat(simi_values_bac).view(-1, 1)
                
                # Attentive Matching:
                wordos_weighted_vecs_for = torch.mul(wordos[emoji_idx, :, 0:self.hidden_dim//2], simi_values_for / simi_values_for.sum())
                wordos_weighted_vecs_bac = torch.mul(wordos[emoji_idx, :, self.hidden_dim//2: ], simi_values_bac / simi_values_bac.sum())
                wordos_weighted_sum_for = torch.sum(wordos_weighted_vecs_for, dim=0).view(1,-1) 
                wordos_weighted_sum_bac = torch.sum(wordos_weighted_vecs_bac, dim=0).view(1,-1) 
                
                m_attentive_for = self.fm(for_emoji_vec, wordos_weighted_sum_for, self.W5).view(1,-1)  # vec5
                m_attentive_bac = self.fm(bac_emoji_vec, wordos_weighted_sum_bac, self.W6).view(1,-1)  # vec6
                
                # Max-Attentive Matching:
                max_value_for, max_idx_for = torch.max(simi_values_for, 0)
                max_value_bac, max_idx_bac = torch.max(simi_values_bac, 0)
                
                max_simi_vec_for = wordos[emoji_idx, int(max_idx_for), 0:self.hidden_dim//2]
                max_simi_vec_bac = wordos[emoji_idx, int(max_idx_bac), self.hidden_dim//2:]
                 
                m_max_attentive_for = self.fm(for_emoji_vec, max_simi_vec_for, self.W7).view(1,-1)  # vec7
                m_max_attentive_bac = self.fm(bac_emoji_vec, max_simi_vec_bac, self.W8).view(1,-1)  # vec8
                
                final_8_vecs = torch.cat([m_full_for, m_full_bac, m_max_for, m_max_bac, 
                                          m_attentive_for, m_attentive_bac, m_max_attentive_for,
                                          m_max_attentive_bac],dim=1)

                emoji_stamps.append(final_8_vecs)
        emoji_stamps = torch.cat(emoji_stamps)
        return emoji_stamps.view(emojios.shape[0], emojios.shape[1], -1)
    
    def cosine_sim(self, v1, v2):
        sim = v1.dot(v2).div(v1.norm(2)*v2.norm(2))
        return sim
    
    def fm(self, v1, v2, W):
        v1s = torch.mul(W, v1)
        v2s = torch.mul(W, v2)
        m_v = torch.cat([self.cosine_sim(v1s[k], v2s[k]) for k in range(v1s.shape[0])])
        return m_v.view(1,-1)        

