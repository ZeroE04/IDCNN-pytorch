# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:22:06 2021

@author: yilin3
"""


import os
import numpy as np
from tqdm import tqdm
from Loss import CRF
import config as config
import torch
import torch.nn as nn
import torch.nn.functional as F

class DICNN_CRF(nn.Module):
    def __init__(self, word_embedding_dim, word2id, num_tag, save_dir, processed_emb,filters=128):
        super(DICNN_CRF, self).__init__()
      
        self.num_tag = num_tag
        self.embedding = nn.Embedding(len(word2id), word_embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(get_embedding(len(word2id),word_embedding_dim,word2id,processed_emb)))
        self.idcnn = IDCNN(emb_dim=word_embedding_dim, filters=filters)
        self.linear = nn.Linear(filters, 256)
        self.out = nn.Linear(256, num_tag)
        self.crf = CRF(num_tags=num_tag, batch_first=True)
        self.save_dir = save_dir

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        out = self.idcnn(embeddings)
        out = self.linear(out)
        out = self.out(out)
        output = F.dropout(out, p=0.1, training=self.training)
        return output
   
    def load(self):
        self.load_state_dict(os.path.join(self.save_dir,config.save_name))

    def save(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.state_dict(), os.path.join(self.save_dir,config.save_name))

    def evaluate_sentence(self, y_pred, y_true):
        all_num = 0 
        correct_num = 0
        if type(y_pred) == list:
            for pred,labeled in zip(y_pred,y_true):
                correct_num += sum([pre.tolist() == lab.tolist() for pre,lab in zip(pred,labeled)])
                all_num += pred.shape[0]
        else:
            correct_num = sum([pred.tolist() == labeled.tolist() for pred,labeled in zip(y_pred,y_true)])
            all_num = y_true.shape[0]
        
        acc = float(correct_num)/all_num
        return acc

class IDCNN(nn.Module):
    def __init__(self, emb_dim, filters, kernel_size=3):
        super(IDCNN, self).__init__()

        self.linear_1 = nn.Linear(emb_dim, filters)
        self.linear_2 = nn.Linear(filters, filters)
        self.linear_3 = nn.Linear(filters, filters)
        self.linear_4 = nn.Linear(filters, filters)

        self.conv_1_1 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_1_2 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_di_1 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=2,padding=kernel_size // 2 + 2 - 1)
        
        self.conv_2_1 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_2_2 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_di_2 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=2,padding=kernel_size // 2 + 2 - 1)

        self.conv_3_1 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_3_2 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_di_3 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=2,padding=kernel_size // 2 + 2 - 1)

        self.conv_4_1 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_4_2 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=1,padding=kernel_size // 2 + 1 - 1)
        self.conv_di_4 = nn.Conv1d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,dilation=2,padding=kernel_size // 2 + 2 - 1)
        
        self.rl = nn.ReLU()
        self.norms = LayerNorm(filters)
        
    def forward(self, embeddings):
        embeddings = nn.Dropout(0.5)(embeddings)
        embeddings = self.linear_1(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        x = self.conv_1_1(embeddings)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_1_2(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_di_1(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.rl(x)
        x = self.norms(x)

        hiddens = x.permute(0, 2, 1)
        hiddens = nn.Dropout(0.1)(hiddens)
        hiddens = self.linear_2(hiddens)
        hiddens = hiddens.permute(0, 2, 1)
        x = self.conv_2_1(hiddens)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_2_2(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_di_2(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.rl(x)
        x = self.norms(x)


        hiddens = x.permute(0, 2, 1)
        hiddens = nn.Dropout(0.1)(hiddens)
        hiddens = self.linear_3(hiddens)
        hiddens = hiddens.permute(0, 2, 1)
        x = self.conv_3_1(hiddens)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_3_2(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_di_3(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.rl(x)
        x = self.norms(x)

        hiddens = x.permute(0, 2, 1)
        hiddens = nn.Dropout(0.1)(hiddens)
        hiddens = self.linear_4(hiddens)
        hiddens = hiddens.permute(0, 2, 1)
        x = self.conv_4_1(hiddens)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_4_2(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.conv_di_4(x)
        x = self.rl(x)
        x = self.norms(x)
        x = self.rl(x)
        x = self.norms(x)

        output = x.permute(0, 2, 1)
        return output

class LayerNorm(nn.Module):
    def __init__(self, filters, elementwise_affine=False):
        super(LayerNorm, self).__init__()
        self.LN = nn.LayerNorm([filters],elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.LN(x)
        return out.permute(0, 2, 1)

def parse_word_vector(word_index,embedding_dim):
    pre_trained_wordvector = {}
    f = open(config.EMBEDDING_FILE, encoding='utf-8')
    fr = f.readlines()
    for line in fr[1:]:
        lines = line.strip().split(' ')
        word = lines[0]
        if len(word)==1:
            if word_index.get(word) is not None:
                vector = [float(f) for f in lines[1:embedding_dim+1]]
                pre_trained_wordvector[word] = vector
            else:
                continue
        else:
            continue
    return pre_trained_wordvector


def get_embedding(vocab_size, embedding_dim, word2id, nil=True):
    print('Get embedding...')
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    if not nil:
        pre_trained_wordector = parse_word_vector(word2id, embedding_dim)
        for word, id in tqdm(word2id.items()):
            try:
                word_vector = pre_trained_wordector[word]
                embedding_matrix[id] = word_vector
            except:
                continue
    print('Get embedding done!')
    return embedding_matrix

