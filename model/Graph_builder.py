#!/usr/bin/python3
'''
@Description: Graph_builder.py
@Version: 0.0
@Autor: wangding
@Date: 2021-01-19-10:36
@Software:PyCharm
LastEditors:  “”
LastEditTime: 2023-12-16 15:24:41
'''

import os
import pickle

import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from tqdm import tqdm,trange
from torch_geometric.data import Data
from Constants import *
from DataConstruct import Options
import random


def BuildRelativePositionEmbedding(max_len,d_model):
    import math
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                        -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

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


# 基于不同关系构建异质图
def BuildRepostGraph(data_name):
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

def BuildItemGraph(data_name):
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

# 基于杰卡德系数构建物品内部关系
def BuildItemInnerGraph(data_name, threshold=0.47):
    import pickle, os, numpy

    options = Options(data_name)

    with open(options.u2idx_dict, 'rb') as handle_1:
        _u2idx = pickle.load(handle_1)
    with open(options.idx2u_dict, 'rb') as handle_2:
        _idx2u = pickle.load(handle_2)
    with open(options.ui2idx_dict, 'rb') as handle_00:
        _ui2idx = pickle.load(handle_00)
    with open(options.idx2ui_dict, 'rb') as handle_0:
        _idx2ui = pickle.load(handle_0)

    all_embedding = []
    with open(data_name + "/cascade.txt", "r") as cascade_file:
        all_cas = [line.split("\n")[0] for line in cascade_file.readlines()]
        for i in range(0, len(all_cas)):
            temp = [0] * len(_idx2u)
            items = all_cas[i].split(" ")
            for item in items:
                if item != "":
                    user, time = item.split(',')
                    temp[_u2idx[user]] = 1.0
            all_embedding.append(temp)

    total = 0
    # if data_name == "./data/memetracker":

    input1 = torch.tensor(all_embedding)
    res = torch.matmul(input1, input1.T)
    # print(res)
    sum_input_1 = torch.sum(input1, dim=1)
    res_list = res.detach().cpu().numpy().tolist()
    sum_list = sum_input_1.detach().cpu().numpy().tolist()
    if os.path.exists(options.item_inner_net_data): os.remove(options.item_inner_net_data)
    with open(options.train_data_id, "r") as id_file:
        with open(options.item_inner_net_data, "a") as file:
            all_id = [line.split("\n")[0] for line in id_file.readlines()]
            for i in trange(0, len(all_id) - 1):
                for j in range(i + 1, len(all_id)):
                    if (res_list[i][j]) != 0:
                        jaccrd_score = (res_list[i][j]) / (sum_list[i] + sum_list[j] - res_list[i][j])
                        if jaccrd_score > threshold:
                            total += 2
                            file.write(f"{all_id[i]},{all_id[j]}\n")
                            file.write(f"{all_id[j]},{all_id[i]}\n")

    # print(total / 2)


# 基于共现构建社交关系，主要针对meme这种用户转发量高而且没有社交图的场景
def BuildSocialGraph(data_name):
    options=Options(data_name)

    import pickle,os
    import numpy as np
    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    user_lists=[[0 for _ in range(len(_idx2ui))] for _ in range(len(_idx2ui))]
    with open(options.train_data) as cas_file:
        with open(options.train_data_id) as id_file:
            id_lines=[line.split("\n")[0] for line in id_file.readlines()]
            cas_lines=[line.split("\n")[0] for line in cas_file.readlines()]
            for i in range(len(cas_lines)):
                line=cas_lines[i]
                id=id_lines[i]
                users=[item.split(",")[0] for item in line.split()]
                for user in users:
                    user_lists[_ui2idx[user]][_ui2idx[id]] = 1
                    user_lists[_ui2idx[id]][_ui2idx[user]] = 1


    print("start counting")
    a=torch.tensor(user_lists,dtype=torch.float32).cuda()
    res= torch.matmul(a,a)
    res_list=res.detach().cpu().numpy().tolist()
    print("finish counting")

    total=0
    if data_name=="./data/memetracker":
        if os.path.exists(options.net_data): os.remove(options.net_data)
        with open(options.net_data,"a") as file:
            for i in range(2,len(_idx2u)-1):
                for j in range(i+1,len(_idx2u)):
                    if res_list[i][j]>10:
                        if int(_idx2u[i]) < 300000 :
                            total+=2
                            file.write(f"{_idx2u[i]},{_idx2u[j]}\n")
                            file.write(f"{_idx2u[j]},{_idx2u[i]}\n")

    print(total/2)

def RefineSocialNetwork(data_name):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    social_user_dict = {}
    with open(options.net_data,"r") as file:
        for line in file.readlines():
            user_1, user_2 = line.split("\n")[0].split(",")
            if user_1 not in social_user_dict.keys():
                social_user_dict[user_1] = []

            social_user_dict[user_1].append(user_2)
            # print(social_user_dict[user_1])
    # print(len(social_user_pair))
    # print(social_user_pair)

    cas_user_dict={}
    with open(options.train_data,"r") as file :
        for line in tqdm(file.readlines()):
            items =line.split("\n")[0].split(" ")
            user= [item.split(",")[0] for item in items]
            for i in range(0,len(user)-1):
                if user[i] not in cas_user_dict.keys():
                    cas_user_dict[user[i]] = []
                cas_user_dict[user[i]]+= user[i:]

    with open(options.valid_data,"r") as file :
        for line in tqdm(file.readlines()):
            items =line.split("\n")[0].split(" ")
            user= [item.split(",")[0] for item in items]
            for i in range(0,len(user)-1):
                if user[i] not in cas_user_dict.keys():
                    cas_user_dict[user[i]] = []
                cas_user_dict[user[i]]+= user[i:]

    with open(options.test_data,"r") as file :
        for line in tqdm(file.readlines()):
            items =line.split("\n")[0].split(" ")
            user= [item.split(",")[0] for item in items]
            for i in range(0,len(user)-1):
                if user[i] not in cas_user_dict.keys():
                    cas_user_dict[user[i]] = []
                cas_user_dict[user[i]]+= user[i:]

    output_user_dict = {}
    for user in cas_user_dict.keys():
        output_user_dict[user]=[]
        for u in cas_user_dict[user]:
            if user in social_user_dict.keys():
                if u in social_user_dict[user] and u not in output_user_dict[user]:
                    output_user_dict[user].append(u)

    if os.path.exists(options.net_data_refined):os.remove(options.net_data_refined)
    with open(options.net_data_refined,"a") as file:
          for user_1 in output_user_dict.keys():
                for user_2 in output_user_dict[user_1]:
                    file.write(f"{user_1},{user_2}\n")


def BuildUserHiddenGraph(data_name, threshold=0.47):
    import pickle, os, numpy

    options = Options(data_name)

    with open(options.u2idx_dict, 'rb') as handle_1:
        _u2idx = pickle.load(handle_1)
    with open(options.idx2u_dict, 'rb') as handle_2:
        _idx2u = pickle.load(handle_2)
    with open(options.ui2idx_dict, 'rb') as handle_00:
        _ui2idx = pickle.load(handle_00)
    with open(options.idx2ui_dict, 'rb') as handle_0:
        _idx2ui = pickle.load(handle_0)

    all_embedding = []
    with open(data_name + "/cascade.txt", "r") as cascade_file:
        all_cas = [line.split("\n")[0] for line in cascade_file.readlines()]
        for i in range(0, len(all_cas)):
            temp = [0] * len(_idx2u)
            items = all_cas[i].split(" ")
            for item in items:
                if item != "":
                    user, time = item.split(',')
                    temp[_u2idx[user]] = 1
            all_embedding.append(temp)

    total = 0
    # if data_name == "./data/memetracker":

    input1 = torch.tensor(all_embedding)
    res = torch.matmul(input1, input1.T)
    # print(res)
    sum_input_1 = torch.sum(input1, dim=1)
    res_list = res.detach().cpu().numpy().tolist()
    sum_list = sum_input_1.detach().cpu().numpy().tolist()
    if os.path.exists(options.user_hidden_net): os.remove(options.user_hidden_net)
    with open(options.train_data_id, "r") as id_file:
        with open(options.user_hidden_net, "a") as file:
            all_id = [line.split("\n")[0] for line in id_file.readlines()]
            for i in trange(0, len(all_id) - 1):
                for j in range(i + 1, len(all_id)):
                    if (res_list[i][j]) != 0:
                        jaccrd_score = (res_list[i][j]) / (sum_list[i] + sum_list[j] - res_list[i][j])
                        if jaccrd_score > threshold:
                            for k in range(len(_idx2u)):
                                if all_embedding[i][k]==0 and all_embedding[j][k]!=0:
                                    file.write(f"{_idx2u[k]},{all_id[i]}\n")
                                elif all_embedding[i][k]!=0 and all_embedding[j][k]==0:
                                    file.write(f"{_idx2u[k]},{all_id[j]}\n")

    # print(total / 2)

# 构建附加项目
def GetTimeEmbedding(data_name="./data/twitter", relative=False):

    options = Options(data_name)
    time_set = set()


    pos_embedding = nn.Embedding(300000,8)

    if data_name=="./data/memetracker": relative=True
    if relative:
        pos_embedding = nn.Embedding.from_pretrained(BuildRelativePositionEmbedding(300000,8))

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
                        time_line.append(float(time))
                        time_set.add(time)

    time_line = sorted(time_line)
    time_line_tensor = torch.tensor(time_line).view(1, -1)

    time_line_embedding = pos_embedding(torch.arange(time_line_tensor.size(1)).expand(time_line_tensor.size()))
    time_line_embedding = time_line_embedding.view(-1, 8)
    time_line_embedding_list = time_line_embedding.cpu().detach().numpy().tolist()

    print(time_line_embedding.size())
    print(len(time_set))

    embedding_dict = {}
    for i in range(0, len(time_line)):
        embedding_dict[str(time_line[i])] = []
        embedding_dict[str(time_line[i])] = time_line_embedding_list[i]

    return embedding_dict

def BuildRepostGraphWithTime(data_name):

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
    if os.path.exists(options.repost_net_data): os.remove(options.repost_net_data)
    if os.path.exists(options.repost_time_data): os.remove(options.repost_time_data)
    with open(options.train_data, "r") as train_data:
        with open(options.repost_net_data, "a") as edge_file:
            with open(options.repost_time_data, "a") as time_file:
                for line in train_data.readlines():
                    edges = []
                    items = [item for item in line.split("\n")[0].strip().split(" ")]
                    for i in range(0, len(items) - 1):
                        edges.append((items[i], items[i + 1]))

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

def BuildRepostGraphWeight(data_name):
    options = Options(data_name)
    _u2idx = {}
    _idx2u = []

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    time_embedding = GetTimeEmbedding(data_name)

    # train_data = open(options.train_data, "r");
    # lines = train_data.readlines()

    if os.path.exists(options.repost_net_data_with_time_weight): os.remove(options.repost_net_data_with_time_weight)

    with open(options.repost_time_data, "r") as input_file:
        with open(options.repost_net_data_with_time_weight, "a") as weight_file:
            lines = [line.split("\n")[0] for line in input_file.readlines()]
            for i in range(0, len(lines)):
                time = lines[i]
                weight_file.write(f"{time_embedding[time]}".replace("[", "").replace("]", "") + "\n")

    # train_data.close()

def BuildItemGraphWeight(data_name):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    time_embedding = GetTimeEmbedding(data_name)

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

def BuildTrimedGraphs(data_name="./data/twitter", item_threshold=3, threshold=3):
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
        if os.path.exists(options.item_net_data_with_time_weight_trimmed): os.remove(options.item_net_data_with_time_weight_trimmed)
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
                        edge_lines = input_edge_file.readlines()
                        weight_lines = input_weight_file.readlines()
                        for i in range(0, len(edge_lines)):
                            line = edge_lines[i]
                            u1, u2 = line.split("\n")[0].split(",")
                            if train_user_dict[u1] > threshold and train_user_dict[u2] > threshold:
                                output_edge_file.write(edge_lines[i])
                                output_weight_file.write(weight_lines[i])


def PreprocessByType(data_name,Type):
    option = Options(data_name)

    if os.path.exists(option.net_data_refined):os.remove(option.net_data_refined)
    if os.path.exists(option.item_net_data):os.remove(option.item_net_data)
    if os.path.exists(option.repost_net_data):os.remove(option.repost_net_data)
    if os.path.exists(option.repost_time_data):os.remove(option.repost_time_data)
    if os.path.exists(option.item_net_data_with_time_weight):os.remove(option.item_net_data_with_time_weight)
    if os.path.exists(option.repost_net_data_with_time_weight):os.remove(option.repost_net_data_with_time_weight)
    if os.path.exists(option.item_net_data_trimmed):os.remove(option.item_net_data_trimmed)
    if os.path.exists(option.item_net_data_with_time_weight_trimmed):os.remove(option.item_net_data_with_time_weight_trimmed)
    if os.path.exists(option.repost_net_data_trimmed):os.remove(option.repost_net_data_trimmed)
    if os.path.exists(option.repost_net_data_with_time_weight_trimmed):os.remove(option.repost_net_data_with_time_weight_trimmed)

    # BuildRepostGraphWithTime(data_name)
    # BuildRepostGraphWeight(data_name)
    # BuildItemGraph(data_name)
    # BuildItemGraphWeight(data_name)

    # if data_name == "./data/twitter":
    #     BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
    #     # BuildUserHiddenGraph(data_name,threshold=0.85)
    # elif data_name == "./data/douban":
    #     BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
    #     # BuildUserHiddenGraph(data_name,threshold=0.6)
    # elif data_name == "./data/memetracker":
    #     BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
    #     # BuildUserHiddenGraph(data_name,threshold=0.8)


# 异质图整合
def LoadHeteStaticGraph(data_name,Type):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    PreprocessByType(data_name,Type)


    edges_list = []
    edges_type_list = []
    edges_weight_list = []

    if os.path.exists(options.repost_net_data) is False:
        BuildRepostGraph(data_name)

    if os.path.exists(options.item_net_data) is False:
        BuildItemGraph(data_name)

    if Type.find("item") != -1:
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
            edges_weight_list+=[1.0] * len(relation_list_reverse)
    
    if Type.find("social") != -1:
        if os.path.exists(options.net_data) is False: print("There exists no social grpah!!")
        else:
            if os.path.exists(options.net_data_refined) is False:
                RefineSocialNetwork(data_name)
            with open(options.net_data_refined, 'r') as handle:
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
                edges_weight_list+=[1.0] * len(relation_list_reverse)
        print("load graph!")

    if Type.find("diffusion") != -1:
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
            edges_weight_list+=[1.0] * len(relation_list_reverse)

    # with open(options.item_inner_net_data, 'r') as handle:
    #     relation_list = handle.read().strip().split("\n")
    #     relation_list = [edge.split(',') for edge in relation_list]
    #     relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
    #                      edge[0] in _ui2idx and edge[1] in _ui2idx]
    #     # print(relation_list)
    #     relation_list_reverse = [edge[::-1] for edge in relation_list]
    #     temp_edges_type_list = [3] * len(relation_list_reverse)
    #     # print(relation_list_reverse)
    #     edges_list += relation_list_reverse
    #     edges_type_list += temp_edges_type_list
    #     edges_weight_list.append([1.0] * len(edges_list))
    # with open(options.user_hidden_net, 'r') as handle:
    #     relation_list = handle.read().strip().split("\n")
    #     relation_list = [edge.split(',') for edge in relation_list]
    #     relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
    #                      edge[0] in _ui2idx and edge[1] in _ui2idx]
    #     # print(relation_list)
    #     relation_list_reverse = [edge[::-1] for edge in relation_list]
    #     temp_edges_type_list = [3] * len(relation_list_reverse)
    #     # print(relation_list_reverse)
    #     edges_list += relation_list_reverse
    #     edges_type_list += temp_edges_type_list
    #     edges_weight_list+=[0.4] * len(edges_list)

    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_type = torch.LongTensor(edges_type_list)
    edges_weight = torch.FloatTensor(edges_weight_list)
    # edges_weight = torch.ones(edges_list_tensor.size(1)).float()

    graph = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_attr=edges_weight)

    return graph


