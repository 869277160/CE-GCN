#!/usr/bin/python3
'''
@Description: Merger.py
@Version: 0.0
@Autor: wangding
@Date: 2020-10-28-8:48
@Software:PyCharm
@LastEditors: wangding
@LastEditTime:  2020-10-28-8:48
'''

import torch.nn as nn
import torch.nn.init as init

from Constants import *

#输入输出相同
class Merger_0(nn.Module):
    def __init__(self,opt,dropout):
        super(Merger_0, self).__init__()
        self.final_dim = opt.d_word_vec

    def forward(self,a):
        return a

#直接拼接之后输出
class Merger_1(nn.Module):
    def __init__(self,opt,dropout):
        super(Merger_1, self).__init__()
        self.final_dim = opt.d_word_vec * 2

    def forward(self,user_dyemb,user_user_dyemb):
        dyemb = torch.cat([user_dyemb,user_user_dyemb], dim=-1).cuda()  # dynamic_node_emb
        return dyemb

# 一层MLP特征筛选
class Merger_2(nn.Module):
    def __init__(self,opt,dropout=0.1,input_dim=128):
        super(Merger_2, self).__init__()

        self.final_dim = 64

        self.linear = nn.Linear(input_dim, self.final_dim)
        init.xavier_normal_(self.linear.weight)

    def forward(self,a,b):
        ui_dyemb=torch.cat([a, b], dim=-1).cuda()
        dyemb=self.linear(ui_dyemb)

        return dyemb

#基于 sigmoid 进行筛选
class Merger_3(nn.Module):
    def __init__(self,opt,dropout):
        super(Merger_3, self).__init__()

        self.final_dim = opt.d_word_vec

        self.linear_a = nn.Linear(64,1)
        init.xavier_normal_(self.linear_a.weight)

        self.linear_b = nn.Linear(64,1)
        init.xavier_normal_(self.linear_b.weight)

    def forward(self,a,b):
        score = nn.functional.sigmoid(self.linear_a(a)+self.linear_b(b))
        user_embedding = score * a + (1 - score) * b
        return user_embedding.cuda()

#DyHGCN方式进行筛选
class Merger_4(nn.Module):
    def __init__(self,opt,dropout):
        super(Merger_4, self).__init__()

        self.final_dim = opt.d_word_vec

        self.linear_dyemb = nn.Linear(64*4, self.final_dim)
        init.xavier_normal_(self.linear_dyemb.weight)

    def forward(self,a,b):

        dyemb = torch.cat([a,b,a * b,a - b], dim=-1).cuda()  # dynamic_node_emb

        dyemb = self.linear_dyemb(dyemb)

        return dyemb
