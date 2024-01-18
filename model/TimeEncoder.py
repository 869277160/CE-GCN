#!/usr/bin/python3
'''
@Description: TimeEncoder.py
@Version: 0.0
@Autor: wangding
@Date: 2021-01-20-16:42
@Software:PyCharm
LastEditors:  “”
LastEditTime: 2023-12-15 17:46:40
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



class TimeEncoder(nn.Module):
    def __init__(self,opt):
        super(TimeEncoder, self).__init__()

        data_name="./data/"+opt.data
        self.pass_time=self.GetPasstime(data_name)
        self.n_time_interval = opt.time_interval
        self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=8
        self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        init.xavier_normal_(self.linear_1.weight)
        self.relu=nn.ReLU()



    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        t_1=timestamp[:,1:]
        t_2=timestamp[:,:-1]
        pass_time=t_1-t_2
        pass_time=pass_time / self.per_time
        pass_time=self.relu(pass_time.floor_().contiguous().view(batch_size*max_len,1).int())

        pass_time=pass_time.long()

        time_embedding_one_hot=torch.zeros(batch_size*max_len, self.n_time_interval)
        time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).cuda()

        time_embedding = self.linear_1(time_embedding_one_hot)

        time_embedding=time_embedding.view(batch_size, max_len, self.output_dim).cuda()

        return time_embedding.cuda(),timestamp[:, :-1]

    def GetPasstime(self,data_name):
            options = Options(data_name)
            max_time = 0
            min_time = 1000000000000
            with open(options.train_data) as file:
                lines = [line.split("\n")[0].strip() for line in file.readlines()]
                for line in lines:
                    times = [item.split(",")[1] for item in line.split()]
                    for time in times:
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            int_time = float(time.ljust(max_len, '0'))
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time
                        else:
                            int_time = float(time)
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time

            with open(options.valid_data) as file:
                lines = [line.split("\n")[0].strip() for line in file.readlines()]
                for line in lines:
                    times = [item.split(",")[1] for item in line.split()]
                    for time in times:
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            int_time = float(time.ljust(max_len, '0'))
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time
                        else:
                            int_time = float(time)
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time

            with open(options.test_data) as file:
                lines = [line.split("\n")[0].strip() for line in file.readlines()]
                for line in lines:
                    times = [item.split(",")[1] for item in line.split()]
                    for time in times:
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            int_time = float(time.ljust(max_len, '0'))
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time
                        else:
                            int_time = float(time)
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time

            # print(max_time)
            # print(min_time)
            # print(max_time - min_time)
            return max_time - min_time


class LinearTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(LinearTimeEncoder, self).__init__()
        data_name="./data/"+opt.data

        # self.n_time_interval = opt.time_interval
        # self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=8
        # self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        # init.xavier_normal_(self.linear_1.weight)
        # self.relu=nn.ReLU()

    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        # t_1=timestamp[:,1:]
        # t_2=timestamp[:,:-1]
        # pass_time=t_1-t_2
        # pass_time=pass_time / self.per_time
        # pass_time=self.relu(pass_time.floor_().contiguous().view(batch_size*max_len,1).int())

        # pass_time=pass_time.long()


        # time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).cuda()

        # time_embedding = self.linear_1(time_embedding_one_hot)

        time_embedding =torch.zeros(batch_size, max_len, self.output_dim).cuda()
        one_time_embedding = self.BuildRelativePositionEmbedding(max_len,self.output_dim)
        time_embedding[:,:,:]=one_time_embedding[:,:]
        # time_embedding=time_embedding.view(
        return time_embedding.cuda(),timestamp[:, :-1]
    
    def BuildRelativePositionEmbedding(self,max_len,d_model):
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

class SlideTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(SlideTimeEncoder, self).__init__()

        data_name="./data/"+opt.data

        self.pass_time,self.min_time=self.GetPasstime(data_name)
        self.n_time_interval = 1000
        self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=8
        self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        init.xavier_normal_(self.linear_1.weight)
        self.relu=nn.ReLU()


    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        # t_1=timestamp[:,1:]
        # t_2=timestamp[:,:-1]
        # pass_time=t_1-t_2
        timestamp=timestamp[:, :-1]
        pass_time=(timestamp-self.min_time)/self.per_time
        
        pass_time=self.relu(pass_time.floor_().contiguous().view(batch_size*max_len,1).int()).long()
 
 
        time_embedding_one_hot=torch.zeros(batch_size*max_len, self.n_time_interval)
        time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).cuda()

        time_embedding = self.linear_1(time_embedding_one_hot)

        time_embedding=time_embedding.view(batch_size, max_len, self.output_dim).cuda()

        return time_embedding.cuda(),timestamp

    def GetPasstime(self,data_name):
            options = Options(data_name)
            max_time = 0
            min_time = 1000000000000
            with open(options.train_data) as file:
                lines = [line.split("\n")[0].strip() for line in file.readlines()]
                for line in lines:
                    times = [item.split(",")[1] for item in line.split()]
                    for time in times:
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            int_time = int(time.ljust(max_len, '0'))
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time
                        else:
                            int_time = int(time)
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time

            with open(options.valid_data) as file:
                lines = [line.split("\n")[0].strip() for line in file.readlines()]
                for line in lines:
                    times = [item.split(",")[1] for item in line.split()]
                    for time in times:
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            int_time = int(time.ljust(max_len, '0'))
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time
                        else:
                            int_time = int(time)
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time

            with open(options.test_data) as file:
                lines = [line.split("\n")[0].strip() for line in file.readlines()]
                for line in lines:
                    times = [item.split(",")[1] for item in line.split()]
                    for time in times:
                        max_len = 10
                        if data_name.find("memetracker") != -1: max_len = 12
                        if len(time) != max_len:
                            int_time = int(time.ljust(max_len, '0'))
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time
                        else:
                            int_time = int(time)
                            if int_time > max_time:
                                max_time = int_time
                            if int_time < min_time:
                                min_time = int_time

            # print(max_time)
            # print(min_time)
            # print(max_time - min_time)
            return max_time - min_time,min_time




class NoneTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(NoneTimeEncoder, self).__init__()

        data_name="./data/"+opt.data

        # self.pass_time=self.GetPasstime(data_name)
        # self.n_time_interval = 100000
        # self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=0
        # self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        # init.xavier_normal_(self.linear_1.weight)
        # self.relu=nn.ReLU()

    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        time_embedding=torch.zeros(batch_size,max_len, self.output_dim).cuda()

        return time_embedding.cuda(),timestamp[:, :-1]



