#!/usr/bin/python3
'''
@Description: UserMerger.py
@Version: 0.0
@Autor: wangding
@Date: 2021-02-04-12:17
@Software:PyCharm
LastEditors: Please set LastEditors
LastEditTime: 2021-02-06 21:13:47
'''

from torch.nn import functional as F
import Constants
import torch
import numpy as np
from torch.autograd import Variable

from model.GNN_embeddings import *
from model.GraphEncoder import GraphEncoder
from model.Merger import *
from model.TransformerBlock import TransformerBlock

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class UserMerger(nn.Module):
    def __init__(self,opt):
        super(UserMerger, self).__init__()

        data_name="./data/"+opt.data
        options=Options(data_name)
        with open(options.u2idx_dict, 'rb') as handle_1:
            _u2idx = pickle.load(handle_1)
        with open(options.idx2u_dict, 'rb') as handle_2:
            _idx2u = pickle.load(handle_2)
        with open(options.ui2idx_dict, 'rb') as handle_00:
            _ui2idx = pickle.load(handle_00)
        with open(options.idx2ui_dict, 'rb') as handle_0:
            _idx2ui = pickle.load(handle_0)

        # self.BuildUserRepostList(data_name)
        # user_repost=[]
        # with open(options.user_repost_file, 'rb') as handle_1:
        #     user_repost= pickle.load(handle_1)
        #
        # self.user_repost_matrix=torch.FloatTensor(user_repost).cuda()
        #
        # self.ntoken=len(_idx2u)
        self.input_dim=opt.d_word_vec
        self.output_dim=opt.d_word_vec
        self.linear_1= nn.Linear(self.input_dim, self.output_dim)
        init.xavier_normal_(self.linear_1.weight)
        self.linear_2= nn.Linear(self.input_dim, self.output_dim)
        init.xavier_normal_(self.linear_2.weight)
        # self.tanh=F.tanh()

    def forward(self,user_embedding,timestamp,train):

        batch_size,max_len,dim=user_embedding.size()
        Q=F.tanh(self.linear_1(user_embedding).cuda())
        K=F.tanh(self.linear_2(user_embedding).cuda())
        Q_K=torch.einsum("bld,bmd->bml",Q,K).cuda()
        temperature = dim ** 0.5
        episilon = 1e-6
        Q_K = Q_K / (temperature + episilon)
        mask = torch.zeros([max_len, max_len]).cuda()
        mask += -2**32+1
        mask = torch.triu(mask, diagonal=0).cuda()

        b_mask=torch.zeros_like(Q_K).cuda()
        b_mask[:,:,:]=mask[:,:]

        Q_K += b_mask
        score = F.softmax(Q_K, dim=-1).cuda()
        output =torch.einsum("bll,bmd->bld",score,user_embedding)

        return output


