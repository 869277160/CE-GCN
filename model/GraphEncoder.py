#!/usr/bin/python3
'''
@Description: GraphEncoder.py
@Version: 0.0
@Autor: wangding
@Date: 2021-01-19-10:14
@Software:PyCharm
LastEditors:  “”
LastEditTime: 2023-12-16 15:22:13
'''
import os
import pickle

import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv, SAGEConv,GATConv
from torch_geometric.data import Data
from Constants import *
from DataConstruct import Options
from model.Graph_builder import LoadHeteStaticGraph


# basic class for graph embedding generater
class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.1):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)

        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        # init.xavier_normal_(self.gnn1.weight)
        # init.xavier_normal_(self.gnn2.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        # print (graph_x_embeddings.shape)
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        return graph_output.cuda()


class GraphEncoder(nn.Module):
    def __init__(self, opt, dropout=0.1,Type=""):
        super(GraphEncoder, self).__init__()
        self.dropedge=opt.dropout
        option = Options(opt.data_path)
        self._ui2idx = {}
        with open(option.ui2idx_dict, 'rb') as handle:
            self._ui2idx = pickle.load(handle)

        self.ntoken = len(self._ui2idx)
        self.ninp = opt.d_word_vec
        self.output_dim=opt.d_word_vec

        self.graph = LoadHeteStaticGraph(opt.data_path,Type)

        self.gnn_layer = GraphNN(self.ntoken, self.ninp)

    def forward(self,input,input_timestamp,train=True):

        batch_size, max_len = input.size()

        # if train:
        #     graph=DropEdge(self.edges_list,self.edges_type_list,self.dropedge)
        # else :

        user_social_embedding_lookup = self.gnn_layer(self.graph).cuda()  # [user_size, user_embedding]

        user_input=input.contiguous().view(batch_size*max_len,1).cuda()
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken).cuda()
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_social_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_social_embedding_lookup).view(batch_size,max_len,self.ninp).cuda()

        return user_social_embedding.cuda()


