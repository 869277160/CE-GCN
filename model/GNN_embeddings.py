# #!/usr/bin/python3
# '''
# @Description: social_embedding.py.py
# @Version: 0.0
# @Autor: wangding
# @Date: 2020-10-27-21:23
# @Software:PyCharm
# @LastEditors: wangding
# @LastEditTime:  2020-10-27-21:23
# '''
#
import os
import pickle

import torch.nn as nn
import torch.nn.init as init
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

from Constants import *
from DataConstruct import Options
import random

def handle_line(line):
    user_lists=[]
    items = line.split("\n")[0].strip().split(" ")
    now_user, systime = items[0].split(",")
    systime = int(systime)
    temp = [items[0]]

    for i in range(1, len(items)):
        user,time = items[i].split(",")
        time = int(time)
        if time - systime <= 1:
            temp.append(items[i])
        else:
            user_lists.append(temp)
            # print(user_lists)
            systime = time
            temp=[items[i]]
    user_lists.append(temp)

    # if len(items) == len(user_lists):
    #     is_trimmed = False
    # else:
    #     is_trimmed = True

    return user_lists#,is_trimmed

def DropEdge(edges_list, edges_type,edges_weight,prob):
    if prob != 0.0:
        new_edges_list = []
        new_edges_type = []
        new_edges_weight = []
        for i in range(0, len(edges_list)):
            droprate = random.randint(0, 99)
            # print(droprate)
            if droprate >= prob * 100:
                new_edges_list.append(edges_list[i])
                new_edges_type.append(edges_type[i])
                new_edges_weight.append(edges_weight[i])
    else :
        new_edges_list = edges_list
        new_edges_type = edges_type
        new_edges_weight = edges_weight

    edges_list_tensor = torch.LongTensor(new_edges_list).t()
    edges_type_tensor = torch.LongTensor(new_edges_type)
    edges_weight_tensor = torch.LongTensor(new_edges_weight)
    # edges_weight_tensor = torch.ones(edges_list_tensor.size(1)).float()

    graph = Data(edge_index=edges_list_tensor,edge_type=edges_type_tensor, edge_attr=edges_weight_tensor)


    return graph

# basic class for graph embedding generater
class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.2):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)

        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        init.xavier_normal_(self.gnn1.weight)
        init.xavier_normal_(self.gnn2.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        # print (graph_x_embeddings.shape)
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        return graph_output.cuda()

class UserItemWithRepostEmbedding(nn.Module):
    def __init__(self, opt, dropout=0.1,GAT=False):
        super(UserItemWithRepostEmbedding, self).__init__()
        self.dropedge=opt.dropout
        option = Options(opt.data_path)
        self._ui2idx = {}
        with open(option.ui2idx_dict, 'rb') as handle:
            self._ui2idx = pickle.load(handle)

        self.ntoken = len(self._ui2idx)
        self.ninp = opt.d_word_vec
        self.output_dim=64

        self.gnn_layer = GraphNN(self.ntoken, self.ninp)

        self.edges_list,self.edges_type_list,self.edges_weight_list=self.LoadHeteStaticGraph(opt.data_path)

        self.graph = DropEdge(self.edges_list, self.edges_type_list,self.edges_weight_list, 0.0)

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

    def BuildRepostGraph(self, data_name):
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()

        with open(options.repost_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for i in range(0, len(items) - 2):
                    user1, _ = items[i].split(",")
                    user2, _ = items[i + 1].split(",")
                    file.write(f"{user1},{user2}\n")

        train_data.close()

    def BuildItemGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

        with open(options.item_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, _ = item.split(",")
                        file.write(f"{user},{ids[i]}\n")

        train_data.close()
        train_data_id.close()

    def LoadHeteStaticGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        edges_list = []
        edges_type_list=[]
        edges_weight_list=[]
        if os.path.exists(options.repost_net_data) is False:
            self.BuildRepostGraph(data_name)

        if os.path.exists(options.item_net_data) is False:
            self.BuildItemGraph(data_name)

        with open(options.item_net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [0] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list

        # with open(options.net_data, 'r') as handle:
        #     relation_list = handle.read().strip().split("\n")
        #     relation_list = [edge.split(',') for edge in relation_list]
        #     relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
        #                      edge[0] in _ui2idx and edge[1] in _ui2idx]
        #     # print(relation_list)
        #     relation_list_reverse = [edge[::-1] for edge in relation_list]
        #     temp_edges_type_list = [1] * len(relation_list_reverse)
        #     # print(relation_list_reverse)
        #     edges_list += relation_list_reverse
        #     edges_type_list += temp_edges_type_list

        with open(options.repost_net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [2] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list

        edges_weight_list=[1.0] * len(edges_list)
        # edges_list_tensor = torch.LongTensor(edges_list).t()
        # edges_type = torch.LongTensor(edges_type_list)
        # edges_weight = torch.ones(edges_list_tensor.size(1)).float()
        #
        # data = Data(edge_index=edges_list_tensor,edge_type=edges_type, edge_attr=edges_weight)
        # # print(data)
        return edges_list,edges_type_list,edges_weight_list

class UserItemWithRepostTimeEmbedding(nn.Module):
    def __init__(self, opt, dropout=0.1,GAT=False):
        super(UserItemWithRepostTimeEmbedding, self).__init__()

        option = Options(opt.data_path)
        self._ui2idx = {}
        with open(option.ui2idx_dict, 'rb') as handle:
            self._ui2idx = pickle.load(handle)

        self.ntoken = len(self._ui2idx)
        self.ninp = opt.d_word_vec
        self.output_dim=64
        self.embedding_dim=8 # 8,12,16,24,32,
        print(f"time_embedding_with:{self.embedding_dim}")

        self.gnn_layer = GraphNN(self.ntoken, self.ninp)

        edges_list,edges_type_list,edges_weight_list=self.LoadHeteStaticGraph(opt.data_path)

        self.graph = DropEdge(edges_list,edges_type_list,edges_weight_list, 0.0)

    def forward(self,input,input_timestamp,train=True):

        batch_size, max_len = input.size()

        # if train:
        #     graph=DropEdge(self.edges_list,self.edges_type_list,self.dropedge)
        # else :
        # graph = DropEdge(self.edges_list, self.edges_type_list,self.edges_weight_list, 0.0)

        user_social_embedding_lookup = self.gnn_layer(self.graph).cuda()  # [user_size, user_embedding]


        user_input=input.contiguous().view(batch_size*max_len,1).cuda()
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken).cuda()
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_social_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_social_embedding_lookup).view(batch_size,max_len,self.ninp).cuda()

        return user_social_embedding.cuda()

    def BuildRepostGraph(self, data_name):
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()

        with open(options.repost_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for i in range(0, len(items) - 2):
                    user1, _ = items[i].split(",")
                    user2, _ = items[i + 1].split(",")
                    file.write(f"{user1},{user2}\n")

        train_data.close()

    def BuildItemGraph(self, data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

        with open(options.item_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, _ = item.split(",")
                        file.write(f"{user},{ids[i]}\n")

        train_data.close()
        train_data_id.close()
        
    def GetTimeEmbedding(self, data_name="./data/twitter",relative=False):
        def BuildRelativePositionEmbedding(max_len,d_model):
            import math
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            return pe
        
        options = Options(data_name)
        time_set = set()


        pos_embedding = nn.Embedding(300000, self.embedding_dim)
        if relative:
            pos_embedding =nn.Embedding.from_pretrained(BuildRelativePositionEmbedding(300000, self.embedding_dim))



        time_line = []
        with open(options.train_data, 'r') as file:
            for line in file.readlines():
                items = line.split(" ")
                for item in items:
                    if item != "\n":
                        user, time = item.split(",")
                        if time not in time_set:
                            time_line.append(int(time))

                            time_set.add(time)

        time_line = sorted(time_line)
        time_line_tensor = torch.tensor(time_line).view(1, -1)

        time_line_embedding = pos_embedding(torch.arange(time_line_tensor.size(1)).expand(time_line_tensor.size()))
        time_line_embedding = time_line_embedding.view(-1, self.embedding_dim)
        time_line_embedding_list = time_line_embedding.cpu().detach().numpy().tolist()

        print(time_line_embedding.size())
        print(len(time_set))

        embedding_dict = {}
        for i in range(0, len(time_line)):
            embedding_dict[str(time_line[i])] = []
            embedding_dict[str(time_line[i])] = time_line_embedding_list[i]

        return embedding_dict

    def BuildRepostGraphWeight(self, data_name):
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        time_embedding= self.GetTimeEmbedding(data_name)
    

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()


        if os.path.exists(options.repost_net_data_with_time_weight): os.remove(options.repost_net_data_with_time_weight)
        with open(options.repost_net_data_with_time_weight,"a") as weight_file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for i in range(0, len(items) - 2):
                    _, _ = items[i].split(",")
                    _, time = items[i + 1].split(",")
                    weight_file.write(f"{time_embedding[time]}".replace("[","").replace("]","")+"\n")

        train_data.close()

    def BuildItemGraphWeight(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        time_embedding = self.GetTimeEmbedding(data_name)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

        if os.path.exists(options.item_net_data_with_time_weight): os.remove(options.item_net_data_with_time_weight)
        with open(options.item_net_data_with_time_weight, "a") as weight_file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, time = item.split(",")
                        weight_file.write(f"{time_embedding[time]}".replace("[", "").replace("]", "") + "\n")

        train_data.close()
        train_data_id.close()

    def LoadHeteStaticGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        edges_list = []
        edges_type_list=[]
        edges_weight_list=[]
        if os.path.exists(options.repost_net_data) is False:
            self.BuildRepostGraph(data_name)
        if os.path.exists(options.item_net_data) is False:
            self.BuildItemGraph(data_name)
        if os.path.exists(options.repost_net_data_with_time_weight) is False:
            self.BuildRepostGraphWeight(data_name)
        if os.path.exists(options.item_net_data_with_time_weight) is False:
            self.BuildItemGraphWeight(data_name)

        with open(options.item_net_data_with_time_weight, 'r') as handle:
            for _ in range(0,len(handle.readlines())):
                edges_weight_list.append([1]*self.embedding_dim)

        with open(options.item_net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [0] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list


        with open(options.repost_net_data_with_time_weight, 'r') as handle:
            for line in handle.readlines():
                times = line.split("\n")[0].strip().split(",")
                time_embedding = [float(time) for time in times]
                edges_weight_list.append(time_embedding)

        with open(options.repost_net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [2] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list


        return edges_list,edges_type_list,edges_weight_list

class UserItemWithRepostEmbeddingTrimmed(nn.Module):
    def __init__(self, opt, dropout=0.1,GAT=False):
        super(UserItemWithRepostEmbeddingTrimmed, self).__init__()
        self.dropedge=0.1
        option = Options(opt.data_path)
        self._ui2idx = {}
        with open(option.ui2idx_dict, 'rb') as handle:
            self._ui2idx = pickle.load(handle)

        self.ntoken = len(self._ui2idx)
        self.ninp = opt.d_word_vec
        self.output_dim=opt.d_word_vec

        self.gnn_layer = GraphNN(self.ntoken, self.ninp)

        self.edges_list, self.edges_type_list, self.edges_weight_list = self.LoadHeteStaticGraph(opt.data_path)

        self.graph = DropEdge(self.edges_list, self.edges_type_list, self.edges_weight_list, 0.0)

    def forward(self,input,input_timestamp,input_id,train=True):

        batch_size, max_len = input.size()

        user_social_embedding_lookup = self.gnn_layer(self.graph).cuda()  # [user_size, user_embedding]

        user_input=input.contiguous().view(batch_size*max_len,1).cuda()
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken).cuda()
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_social_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_social_embedding_lookup).view(batch_size,max_len,self.ninp).cuda()

        return user_social_embedding.cuda()

    def BuildRepostGraph(self, data_name):
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        lines=[]
        train_data = open(options.train_data, "r")
        lines += train_data.readlines()


        with open(options.repost_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for i in range(0, len(items) - 2):
                    user1, _ = items[i].split(",")
                    user2, _ = items[i + 1].split(",")
                    file.write(f"{user1},{user2}\n")

        train_data.close()

    def BuildItemGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

        with open(options.item_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, _ = item.split(",")
                        file.write(f"{user},{ids[i]}\n")

        train_data.close()
        train_data_id.close()

        # val_data = open(options.valid_data, "r");
        # lines = val_data.readlines()
        # valid_data_id = open(options.valid_data_id, "r");
        # ids = [line.split("\n")[0] for line in valid_data_id.readlines()]
        #
        # with open(options.item_net_data, "a") as file:
        #     for i in range(0, len(lines)):
        #         items = lines[i].split()
        #         for item in items:
        #             if item is not "\n":
        #                 user, _ = item.split(",")
        #                 file.write(f"{user},{ids[i]}\n")
        #
        # val_data.close()
        # valid_data_id.close()
        #
        # test_data = open(options.test_data, "r");
        # lines = test_data.readlines()
        # test_data_id = open(options.test_data_id, "r");
        # ids = [line.split("\n")[0] for line in test_data_id.readlines()]

        # with open(options.item_net_data, "a") as file:
        #     for i in range(0, len(lines)):
        #         items = lines[i].split()
        #         for item in items:
        #             if item is not "\n":
        #                 user, _ = item.split(",")
        #                 file.write(f"{user},{ids[i]}\n")
        #
        # test_data.close()
        # test_data_id.close()

    def BuildTrimedGraphs(self,data_name="./data/twitter", item_threshold=3, threshold=3):
        options = Options(data_name)

        train_user_dict = {}
        with open(options.train_data, "r") as file:
            for line in file.readlines():
                for item in line.split(" "):
                    if item != "\n":
                        user, time = item.split(',')

                        if user not in train_user_dict.keys():
                            train_user_dict[user] = 1
                        else:
                            train_user_dict[user] += 1
        total = 0
        for user in train_user_dict.keys():
            if train_user_dict[user] <= threshold:
                total += 1


        if os.path.exists(options.item_net_data_trimmed): os.remove(options.item_net_data_trimmed)
        with open(options.item_net_data, "r") as input_file:
            with open(options.item_net_data_trimmed, "a") as output_file:
                for line in input_file.readlines():
                    u1, u2 = line.split("\n")[0].split(",")
                    if train_user_dict[u1] > item_threshold:
                        output_file.write(line)

        if os.path.exists(options.repost_net_data_trimmed): os.remove(options.repost_net_data_trimmed)
        with open(options.repost_net_data, "r") as input_file:
            with open(options.repost_net_data_trimmed, "a") as output_file:
                for line in input_file.readlines():
                    u1, u2 = line.split("\n")[0].split(",")
                    if train_user_dict[u1] > threshold and train_user_dict[u2]>threshold:
                        output_file.write(line)

    def LoadHeteStaticGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        edges_list = []
        edges_type_list=[]
        if os.path.exists(options.repost_net_data) is False:
            self.BuildRepostGraph(data_name)

        if os.path.exists(options.item_net_data) is False:
            self.BuildItemGraph(data_name)

        self.BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)

        with open(options.item_net_data_trimmed, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [0] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list

        with open(options.repost_net_data_trimmed, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [1] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list

        edge_weight_list=[1.0] * len(edges_list)

        # edges_list_tensor = torch.LongTensor(edges_list).t()
        # edges_type = torch.LongTensor(edges_type_list)
        # edges_weight = torch.ones(edges_list_tensor.size(1)).float()
        #
        # data = Data(edge_index=edges_list_tensor,edge_type=edges_type, edge_attr=edges_weight)
        # # print(data)

        return edges_list,edges_type_list,edge_weight_list

class UserItemWithRepostTrimmedTimeEmbeddingWithExtraEdge(nn.Module):
    def __init__(self, opt, dropout=0.1,GAT=False):
        super(UserItemWithRepostTrimmedTimeEmbeddingWithExtraEdge, self).__init__()
        # self.dropedge=0.1
        option = Options(opt.data_path)
        self._ui2idx = {}
        with open(option.ui2idx_dict, 'rb') as handle:
            self._ui2idx = pickle.load(handle)

        self.ntoken = len(self._ui2idx)
        self.ninp = opt.d_word_vec
        self.output_dim=opt.d_word_vec
        self.embedding_dim= 8  # 8,12,16,24,32,
        print(f"time_embedding_with:{self.embedding_dim}")

        self.gnn_layer = GraphNN(self.ntoken, self.ninp)

        self.edges_list, self.edges_type_list, self.edges_weight_list = self.LoadHeteStaticGraph(opt.data_path)

        self.graph = DropEdge(self.edges_list, self.edges_type_list, self.edges_weight_list, 0.0)

    def forward(self,input,input_timestamp,input_id,train=True):

        batch_size, max_len = input.size()

        user_social_embedding_lookup = self.gnn_layer(self.graph).cuda()  # [user_size, user_embedding]

        user_input=input.contiguous().view(batch_size*max_len,1).cuda()
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken).cuda()
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_social_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_social_embedding_lookup).view(batch_size,max_len,self.ninp).cuda()

        return user_social_embedding.cuda()

    def GetTimeEmbedding(self, data_name="./data/twitter",relative=False):
        options = Options(data_name)
        time_set = set()

        pos_embedding = nn.Embedding(300000, self.embedding_dim)

        if relative:
            pos_embedding =nn.Embedding.from_pretrained(BuildRelativePositionEmbedding(300000, self.embedding_dim))



        time_line = []
        with open(options.train_data, 'r') as file:
            for line in file.readlines():
                items = line.split("\n")[0].strip().split(" ")
                for item in items:
                    if item != "\n":
                        user, time = item.split(",")
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            time = time.ljust(max_len, '0')
                        if time not in time_set:
                            time_line.append(int(time))
                            time_set.add(time)

        time_line = sorted(time_line)
        time_line_tensor = torch.tensor(time_line).view(1, -1)

        time_line_embedding = pos_embedding(torch.arange(time_line_tensor.size(1)).expand(time_line_tensor.size()))
        time_line_embedding = time_line_embedding.view(-1, self.embedding_dim)
        time_line_embedding_list = time_line_embedding.cpu().detach().numpy().tolist()

        print(time_line_embedding.size())
        print(len(time_set))

        embedding_dict = {}
        for i in range(0, len(time_line)):
            embedding_dict[str(time_line[i])] = []
            embedding_dict[str(time_line[i])] = time_line_embedding_list[i]

        return embedding_dict

    def BuildRepostGraphWithTime(self,data_name):
        
        def GetEdges(user_list_1, user_list_2):
            from itertools import product

            edges = product(user_list_1, user_list_2)

            return edges
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        # with open(options.u2idx_dict, 'rb') as handle:
        #     _u2idx = pickle.load(handle)
        # with open(options.idx2u_dict, 'rb') as handle:
        #     _idx2u = pickle.load(handle)
        if os.path.exists(options.repost_net_data):os.remove(options.repost_net_data)
        if os.path.exists(options.repost_time_data):os.remove(options.repost_time_data)
        with open(options.train_data, "r") as train_data:
            with open(options.repost_net_data, "a") as edge_file:
                  with open(options.repost_time_data, "a") as time_file:
                        for line in train_data.readlines():
                            edges=[]
                            # items = [item for item in line.split("\n")[0].strip().split(" ")]
                            # for i in range(0,len(items)-1):
                            #     edges.append((items[i],items[i+1]))

                            user_lists = handle_line(line)
                            for i in range(0, len(user_lists)-1):
                                edge_list=GetEdges(user_lists[i],user_lists[i+1])
                                for edge in edge_list:
                                    if edge not in edges:
                                        edges.append(edge)

                            for edge in edges:
                                u1, t1 = edge[0].split(",")
                                u2, t2 = edge[1].split(",")
                                edge_file.write(f"{u1},{u2}\n")
                                max_len = 10
                                if data_name.find("memetracker") != -1: max_len = 12
                                if len(t2) != max_len:
                                    t2 = t2.ljust(max_len, '0')
                                time_file.write(f"{t2}\n")

    def BuildItemGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

        if os.path.exists(options.item_net_data):os.remove(options.item_net_data)
        with open(options.item_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, _ = item.split(",")
                        file.write(f"{user},{ids[i]}\n")

        train_data.close()
        train_data_id.close()

    def BuildRepostGraphWeight(self, data_name):
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        time_embedding= self.GetTimeEmbedding(data_name)

        # train_data = open(options.train_data, "r");
        # lines = train_data.readlines()

        if os.path.exists(options.repost_net_data_with_time_weight): os.remove(options.repost_net_data_with_time_weight)

        with open(options.repost_time_data,"r") as input_file:
            with open(options.repost_net_data_with_time_weight,"a") as weight_file:
                lines=[line.split("\n")[0] for line in input_file.readlines()]
                for i in range(0, len(lines)):
                    time = lines[i]
                    weight_file.write(f"{time_embedding[time]}".replace("[","").replace("]","")+"\n")

        # train_data.close()

    def BuildItemGraphWeight(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        time_embedding = self.GetTimeEmbedding(data_name)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

     
        if os.path.exists(options.item_net_data_with_time_weight): os.remove(options.item_net_data_with_time_weight) 
        with open(options.item_net_data_with_time_weight, "a") as weight_file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, time = item.split(",")
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            time = time.ljust(max_len, '0')
                        weight_file.write(f"{time_embedding[time]}".replace("[", "").replace("]", "") + "\n")

        train_data.close()
        train_data_id.close()

    def BuildTrimedGraphs(self,data_name="./data/twitter", item_threshold=3, threshold=3):
        options = Options(data_name)

        train_user_dict = {}
        with open(options.train_data, "r") as file:
            for line in file.readlines():
                for item in line.split(" "):
                    if item != "\n":
                        user, time = item.split(',')

                        if user not in train_user_dict.keys():
                            train_user_dict[user] = 1
                        else:
                            train_user_dict[user] += 1

        if os.path.exists(options.item_net_data_trimmed): os.remove(options.item_net_data_trimmed)
        if os.path.exists(options.item_net_data_with_time_weight_trimmed):os.remove(options.item_net_data_with_time_weight_trimmed)
        with open(options.item_net_data, "r") as input_edge_file:
            with open(options.item_net_data_trimmed, "a") as output_edge_file:
                with open(options.item_net_data_with_time_weight, "r") as input_weight_file:
                    with open(options.item_net_data_with_time_weight_trimmed, "a") as output_weight_file:
                        edge_lines = input_edge_file.readlines()
                        weight_lines = input_weight_file.readlines()
                        for i in range(0, len(edge_lines)):
                            line = edge_lines[i]
                            
                            u1, u2 = line.split("\n")[0].split(",")
                            if train_user_dict[u1] > item_threshold:
                                output_edge_file.write(edge_lines[i])
                                output_weight_file.write(weight_lines[i])



        if os.path.exists(options.repost_net_data_trimmed): os.remove(options.repost_net_data_trimmed)
        if os.path.exists(options.repost_net_data_with_time_weight_trimmed): os.remove(options.repost_net_data_with_time_weight_trimmed)
        with open(options.repost_net_data, "r") as input_edge_file:
            with open(options.repost_net_data_trimmed, "a") as output_edge_file:
                with open(options.repost_net_data_with_time_weight, "r") as input_weight_file:
                    with open(options.repost_net_data_with_time_weight_trimmed, "a") as output_weight_file:
                        edge_lines=input_edge_file.readlines()
                        weight_lines=input_weight_file.readlines()
                        for i in range(0,len(edge_lines)):
                            line=edge_lines[i]
                            u1, u2 = line.split("\n")[0].split(",")
                            if train_user_dict[u1] > threshold and train_user_dict[u2]>threshold:
                                output_edge_file.write(edge_lines[i])
                                output_weight_file.write(weight_lines[i])

    def LoadHeteStaticGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        edges_list = []
        edges_type_list=[]
        edges_weight_list=[]


        self.BuildRepostGraphWithTime(data_name)
        self.BuildItemGraph(data_name)

        
        self.BuildRepostGraphWeight(data_name)
        self.BuildItemGraphWeight(data_name)
    
        if data_name == "./data/twitter" :
            self.BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
        elif data_name == "./data/douban" :
            self.BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
        elif data_name=="./data/memetracker":
            self.BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)

        with open(options.item_net_data_with_time_weight_trimmed, 'r') as handle:
            for _ in range(0,len(handle.readlines())):
                edges_weight_list.append([1]*self.embedding_dim)

            # for line in handle.readlines():
            #     times = line.split("\n")[0].strip().split(",")
            #     time_embedding = [float(time) for time in times]
            #     edges_weight_list.append(time_embedding)

        with open(options.item_net_data_trimmed, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [0] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list

        with open(options.repost_net_data_with_time_weight_trimmed, 'r') as handle:
            for line in handle.readlines():
                times = line.split("\n")[0].strip().split(",")
                time_embedding = [float(time) for time in times]
                edges_weight_list.append(time_embedding)

        with open(options.repost_net_data_trimmed, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [1] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list



        # edges_list_tensor = torch.LongTensor(edges_list).t()
        # edges_type = torch.LongTensor(edges_type_list)
        # edges_weight = torch.ones(edges_list_tensor.size(1)).float()
        #
        # data = Data(edge_index=edges_list_tensor,edge_type=edges_type, edge_attr=edges_weight)
        # # print(data)

        return edges_list,edges_type_list,edges_weight_list

class UserItemWithRepostTrimmedTimeEmbedding(nn.Module):
    def __init__(self, opt, dropout=0.1,GAT=False):
        super(UserItemWithRepostTrimmedTimeEmbedding, self).__init__()
        # self.dropedge=0.1
        option = Options(opt.data_path)
        self._ui2idx = {}
        with open(option.ui2idx_dict, 'rb') as handle:
            self._ui2idx = pickle.load(handle)

        self.ntoken = len(self._ui2idx)
        self.ninp = opt.d_word_vec
        self.output_dim=opt.d_word_vec
        self.embedding_dim= 8  # 8,12,16,24,32,
        print(f"time_embedding_with:{self.embedding_dim}")

        self.gnn_layer = GraphNN(self.ntoken, self.ninp)

        self.edges_list, self.edges_type_list, self.edges_weight_list = self.LoadHeteStaticGraph(opt.data_path)

        self.graph = DropEdge(self.edges_list, self.edges_type_list, self.edges_weight_list, 0.0)

    def forward(self,input,input_timestamp,input_id,train=True):

        batch_size, max_len = input.size()

        user_social_embedding_lookup = self.gnn_layer(self.graph).cuda()  # [user_size, user_embedding]

        user_input=input.contiguous().view(batch_size*max_len,1).cuda()
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken).cuda()
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_social_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_social_embedding_lookup).view(batch_size,max_len,self.ninp).cuda()

        return user_social_embedding.cuda()

    def GetTimeEmbedding(self, data_name="./data/twitter",relative=False):
        
        options = Options(data_name)
        time_set = set()

        pos_embedding = nn.Embedding(300000, self.embedding_dim)

        if relative:
            pos_embedding =nn.Embedding.from_pretrained(BuildRelativePositionEmbedding(300000, self.embedding_dim))



        time_line = []
        with open(options.train_data, 'r') as file:
            for line in file.readlines():
                items = line.split("\n")[0].strip().split(" ")
                for item in items:
                    if item != "\n":
                        user, time = item.split(",")
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            time = time.ljust(max_len, '0')
                        if time not in time_set:
                            time_line.append(int(time))
                            time_set.add(time)

        time_line = sorted(time_line)
        time_line_tensor = torch.tensor(time_line).view(1, -1)

        time_line_embedding = pos_embedding(torch.arange(time_line_tensor.size(1)).expand(time_line_tensor.size()))
        time_line_embedding = time_line_embedding.view(-1, self.embedding_dim)
        time_line_embedding_list = time_line_embedding.cpu().detach().numpy().tolist()

        print(time_line_embedding.size())
        print(len(time_set))

        embedding_dict = {}
        for i in range(0, len(time_line)):
            embedding_dict[str(time_line[i])] = []
            embedding_dict[str(time_line[i])] = time_line_embedding_list[i]

        return embedding_dict

    def BuildRepostGraphWithTime(self,data_name):
        
        def GetEdges(user_list_1, user_list_2):
            from itertools import product

            edges = product(user_list_1, user_list_2)

            return edges
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        # with open(options.u2idx_dict, 'rb') as handle:
        #     _u2idx = pickle.load(handle)
        # with open(options.idx2u_dict, 'rb') as handle:
        #     _idx2u = pickle.load(handle)
        if os.path.exists(options.repost_net_data):os.remove(options.repost_net_data)
        if os.path.exists(options.repost_time_data):os.remove(options.repost_time_data)
        with open(options.train_data, "r") as train_data:
            with open(options.repost_net_data, "a") as edge_file:
                  with open(options.repost_time_data, "a") as time_file:
                        for line in train_data.readlines():
                            edges=[]
                            items = [item for item in line.split("\n")[0].strip().split(" ")]
                            for i in range(0,len(items)-1):
                                edges.append((items[i],items[i+1]))

                            # user_lists = handle_line(line)
                            # for i in range(0, len(user_lists)-1):
                            #     edge_list=GetEdges(user_lists[i],user_lists[i+1])
                            #     for edge in edge_list:
                            #         if edge not in edges:
                            #             edges.append(edge)

                            for edge in edges:
                                u1, t1 = edge[0].split(",")
                                u2, t2 = edge[1].split(",")
                                edge_file.write(f"{u1},{u2}\n")
                                max_len = 10
                                if data_name.find("memetracker") != -1: max_len = 12
                                if len(t2) != max_len:
                                    t2 = t2.ljust(max_len, '0')
                                time_file.write(f"{t2}\n")

    def BuildItemGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

        if os.path.exists(options.item_net_data):os.remove(options.item_net_data)
        with open(options.item_net_data, "a") as file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, _ = item.split(",")
                        file.write(f"{user},{ids[i]}\n")

        train_data.close()
        train_data_id.close()

    def BuildRepostGraphWeight(self, data_name):
        options = Options(data_name)
        _u2idx = {}
        _idx2u = []

        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        time_embedding= self.GetTimeEmbedding(data_name)

        # train_data = open(options.train_data, "r");
        # lines = train_data.readlines()

        if os.path.exists(options.repost_net_data_with_time_weight): os.remove(options.repost_net_data_with_time_weight)

        with open(options.repost_time_data,"r") as input_file:
            with open(options.repost_net_data_with_time_weight,"a") as weight_file:
                lines=[line.split("\n")[0] for line in input_file.readlines()]
                for i in range(0, len(lines)):
                    time = lines[i]
                    weight_file.write(f"{time_embedding[time]}".replace("[","").replace("]","")+"\n")

        # train_data.close()

    def BuildItemGraphWeight(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        time_embedding = self.GetTimeEmbedding(data_name)

        train_data = open(options.train_data, "r");
        lines = train_data.readlines()
        train_data_id = open(options.train_data_id, "r");
        ids = [line.split("\n")[0] for line in train_data_id.readlines()]

     
        if os.path.exists(options.item_net_data_with_time_weight): os.remove(options.item_net_data_with_time_weight) 
        with open(options.item_net_data_with_time_weight, "a") as weight_file:
            for i in range(0, len(lines)):
                items = lines[i].split()
                for item in items:
                    if item is not "\n":
                        user, time = item.split(",")
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            time = time.ljust(max_len, '0')
                        weight_file.write(f"{time_embedding[time]}".replace("[", "").replace("]", "") + "\n")

        train_data.close()
        train_data_id.close()

    def BuildTrimedGraphs(self,data_name="./data/twitter", item_threshold=3, threshold=3):
        options = Options(data_name)

        train_user_dict = {}
        with open(options.train_data, "r") as file:
            for line in file.readlines():
                for item in line.split(" "):
                    if item != "\n":
                        user, time = item.split(',')

                        if user not in train_user_dict.keys():
                            train_user_dict[user] = 1
                        else:
                            train_user_dict[user] += 1

        if os.path.exists(options.item_net_data_trimmed): os.remove(options.item_net_data_trimmed)
        if os.path.exists(options.item_net_data_with_time_weight_trimmed):os.remove(options.item_net_data_with_time_weight_trimmed)
        with open(options.item_net_data, "r") as input_edge_file:
            with open(options.item_net_data_trimmed, "a") as output_edge_file:
                with open(options.item_net_data_with_time_weight, "r") as input_weight_file:
                    with open(options.item_net_data_with_time_weight_trimmed, "a") as output_weight_file:
                        edge_lines = input_edge_file.readlines()
                        weight_lines = input_weight_file.readlines()
                        for i in range(0, len(edge_lines)):
                            line = edge_lines[i]
                            
                            u1, u2 = line.split("\n")[0].split(",")
                            if train_user_dict[u1] > item_threshold:
                                output_edge_file.write(edge_lines[i])
                                output_weight_file.write(weight_lines[i])



        if os.path.exists(options.repost_net_data_trimmed): os.remove(options.repost_net_data_trimmed)
        if os.path.exists(options.repost_net_data_with_time_weight_trimmed): os.remove(options.repost_net_data_with_time_weight_trimmed)
        with open(options.repost_net_data, "r") as input_edge_file:
            with open(options.repost_net_data_trimmed, "a") as output_edge_file:
                with open(options.repost_net_data_with_time_weight, "r") as input_weight_file:
                    with open(options.repost_net_data_with_time_weight_trimmed, "a") as output_weight_file:
                        edge_lines=input_edge_file.readlines()
                        weight_lines=input_weight_file.readlines()
                        for i in range(0,len(edge_lines)):
                            line=edge_lines[i]
                            u1, u2 = line.split("\n")[0].split(",")
                            if train_user_dict[u1] > threshold and train_user_dict[u2]>threshold:
                                output_edge_file.write( edge_lines[i])
                                output_weight_file.write(weight_lines[i])

    def LoadHeteStaticGraph(self,data_name):
        options = Options(data_name)
        _ui2idx = {}
        _idx2ui = []

        with open(options.ui2idx_dict, 'rb') as handle:
            _ui2idx = pickle.load(handle)
        with open(options.idx2ui_dict, 'rb') as handle:
            _idx2ui = pickle.load(handle)

        edges_list = []
        edges_type_list=[]
        edges_weight_list=[]


        self.BuildRepostGraphWithTime(data_name)
        self.BuildItemGraph(data_name)

        
        self.BuildRepostGraphWeight(data_name)
        self.BuildItemGraphWeight(data_name)
    
        if data_name == "./data/twitter" :
            self.BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
        elif data_name == "./data/douban" :
            self.BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
        elif data_name=="./data/memetracker":
            self.BuildTrimedGraphs(data_name, item_threshold=5, threshold=5)

        with open(options.item_net_data_with_time_weight_trimmed, 'r') as handle:
            for _ in range(0,len(handle.readlines())):
                edges_weight_list.append([1]*self.embedding_dim)

            # for line in handle.readlines():
            #     times = line.split("\n")[0].strip().split(",")
            #     time_embedding = [float(time) for time in times]
            #     edges_weight_list.append(time_embedding)

        with open(options.item_net_data_trimmed, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [0] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list

        with open(options.repost_net_data_with_time_weight_trimmed, 'r') as handle:
            for line in handle.readlines():
                times = line.split("\n")[0].strip().split(",")
                time_embedding = [float(time) for time in times]
                edges_weight_list.append(time_embedding)

        with open(options.repost_net_data_trimmed, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [1] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list

        # with open(options.item_inner_data, 'r') as handle:
        #     for _ in range(0,len(handle.readlines())):
        #         edges_weight_list.append([1]*self.embedding_dim)
        #
        # with open(options.item_inner_data, 'r') as handle:
        #     relation_list = handle.read().strip().split("\n")
        #     relation_list = [edge.split(',') for edge in relation_list]
        #     relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
        #                      edge[0] in _ui2idx and edge[1] in _ui2idx]
        #     # print(relation_list)
        #     relation_list_reverse = [edge[::-1] for edge in relation_list]
        #     temp_edges_type_list = [1.0] * len(relation_list_reverse)
        #     # print(relation_list_reverse)
        #     edges_list += relation_list_reverse
        #     edges_type_list += temp_edges_type_list
        #


        # edges_list_tensor = torch.LongTensor(edges_list).t()
        # edges_type = torch.LongTensor(edges_type_list)
        # edges_weight = torch.ones(edges_list_tensor.size(1)).float()
        #
        # data = Data(edge_index=edges_list_tensor,edge_type=edges_type, edge_attr=edges_weight)
        # # print(data)

        return edges_list,edges_type_list,edges_weight_list
