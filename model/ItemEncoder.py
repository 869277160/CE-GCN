#!/usr/bin/python3
'''
@Description: TimeEncoder.py
@Version: 0.0
@Autor: wangding
@Date: 2021-01-20-16:42
@Software:PyCharm
@LastEditors: wangding
@LastEditTime:  2021-01-20-16:42
'''



import Constants
import torch
import numpy as np
from torch.autograd import Variable

from model.GNN_embeddings import *
from model.GraphEncoder import GraphEncoder
from model.Merger import *
from model.TransformerBlock import TransformerBlock

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ItemEncoder(nn.Module):
    def __init__(self,opt):
        super(ItemEncoder, self).__init__()

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

        self.BuildUserRepostList(data_name)
        user_repost=[]
        with open(options.user_repost_file, 'rb') as handle_1:
            user_repost= pickle.load(handle_1)

        self.user_repost_matrix=torch.FloatTensor(user_repost).cuda()

        self.ntoken=len(_idx2u)
        self.input_dim=len(_idx2ui)-len(_idx2u)
        self.output_dim=8
        self.linear_1= nn.Linear(self.input_dim, self.output_dim, bias=True)
        init.xavier_normal_(self.linear_1.weight)

    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        user_input = input.contiguous().view(batch_size * max_len, 1).cuda()
        user_embedding_one_hot = torch.zeros(batch_size * max_len, self.ntoken).cuda()
        user_embedding_one_hot = user_embedding_one_hot.scatter_(1, user_input, 1)

        user_item_one_hot=torch.einsum("bt,td->bd",user_embedding_one_hot,self.user_repost_matrix)

        historical_item_embedding = self.linear_1(user_item_one_hot)

        historical_item_embedding = historical_item_embedding.view(batch_size, max_len, self.output_dim).cuda()

        return historical_item_embedding


    def BuildUserRepostList(self,data_name):
        options = Options(data_name)

        import pickle
        with open(options.u2idx_dict, 'rb') as handle_1:
            _u2idx = pickle.load(handle_1)
        with open(options.idx2u_dict, 'rb') as handle_2:
            _idx2u = pickle.load(handle_2)
        with open(options.ui2idx_dict, 'rb') as handle_00:
            _ui2idx = pickle.load(handle_00)
        with open(options.idx2ui_dict, 'rb') as handle_0:
            _idx2ui = pickle.load(handle_0)

        user_repost_list = [[0 for _ in range(len(_idx2ui) - len(_idx2u))] for _ in range(len(_idx2u))]
        with open(options.train_data, "r") as file:
            with open(options.train_data_id, "r") as id_file:
                casline = [line.split("\n")[0].strip() for line in file.readlines()]
                idline = [line.split("\n")[0].strip() for line in id_file.readlines()]
                for i in range(len(idline)):
                    items = casline[i].split(" ")
                    users = [item.split(",")[0] for item in items]
                    for user in users:
                        user_repost_list[_u2idx[user]][_ui2idx[idline[i]] - len(_idx2u)] = 1
        # print(user_repost_list[2])

        with open(options.user_repost_file, 'wb') as handle:
            pickle.dump(user_repost_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
