from torch.nn.init import zeros_
import Constants
import torch
import numpy as np
from torch.autograd import Variable

from model.GNN_embeddings import *
from model.GraphEncoder import GraphEncoder
from model.TimeEncoder import *
from model.ItemEncoder import ItemEncoder
from model.UserMerger import UserMerger
from model.Merger import *
from model.TransformerBlock import TransformerBlock

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CEGCN(nn.Module):
    def __init__(self,opt):
        super(CEGCN, self).__init__()
        dropout=opt.dropout
        self.ntoken = opt.user_size
        self.ninp = opt.d_word_vec
        self.user_size = self.ntoken
        self.pos_dim = 8
        self.__name__ = "CEGCN"

        self.dropout = nn.Dropout(dropout)
        self.drop_timestamp = nn.Dropout(dropout)
        self.Type = opt.notes
        self.time_encoder_str=opt.time_encoder

        self.time_encoder= TimeEncoder(opt)
        if self.time_encoder_str == "Linear":
            self.time_encoder= LinearTimeEncoder(opt)
        elif self.time_encoder_str == "Slide":
            self.time_encoder= SlideTimeEncoder(opt)
        elif self.time_encoder_str == "None":
            self.time_encoder= NoneTimeEncoder(opt)

        self.user_merger=UserMerger(opt)

        self.encoder = GraphEncoder(opt,opt.dropout,self.Type)
        self.final_dim = self.ninp  # 最终的结果

        self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        self.decoder = TransformerBlock(input_size=self.final_dim + self.pos_dim+self.time_encoder.output_dim, n_heads=8)
        self.linear_1 = nn.Linear(self.final_dim + self.pos_dim+self.time_encoder.output_dim, self.ntoken)

        self.sig=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.linear_2 = nn.Linear(self.ninp, self.ninp,bias=True)
        self.linear_3= nn.Linear(self.ninp, self.ninp,bias=True)
        self.linear_4= nn.Linear(self.ninp+self.ninp, self.ninp,bias=True)
        self.linear_5= nn.Linear(self.ninp*4, self.ninp,bias=True)

        self.init_weights()
        print(self)

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear_1.weight)
        init.xavier_normal_(self.linear_2.weight)
        init.xavier_normal_(self.linear_3.weight)
        init.xavier_normal_(self.linear_4.weight)

    def forward(self, input, input_timestamp, input_id,train=True):
        input = input[:, :-1]  # [bsz,max_len]   input_id:[batch_size]

        # timeencoder
        time_embedding,input_timestamp = self.time_encoder(input, input_timestamp,train)
        # item_embedding=self.item_encoder(input,input_timestamp,train)

        #encoder
        user_embedding = self.encoder(input, input_timestamp,train)
        user_att = self.user_merger(user_embedding, input_timestamp,train)
        # user_embedding = user_att
        # user_embedding = self.dropout(user_embedding)

        # score=self.sig(self.linear_2(user_embedding)+self.linear_3(user_att))
        # user_embedding=score*user_embedding+(1-score)*user_att


        user_embedding = torch.cat([user_embedding, user_att], dim=-1)
        user_embedding = self.tanh(self.linear_4(user_embedding))

        # user_embedding = torch.cat([user_embedding, user_att, user_embedding * user_att, user_embedding - user_att], dim=-1)
        # user_embedding = self.tanh(self.linear_5(user_embedding))


        user_embedding = self.dropout(user_embedding)
        if self.time_encoder_str != "None":
            user_embedding=torch.cat([user_embedding,time_embedding],dim=-1)

        mask = (input == Constants.PAD)
        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))

        final_input = torch.cat([user_embedding, order_embed], dim=-1).cuda()  # dynamic_node_emb
        att_out = self.decoder(final_input.cuda(), final_input.cuda(), final_input.cuda(), mask=mask.cuda())
        att_out = self.dropout(att_out.cuda())
        # print(att_out.size())   # [bsz,max_len,pos+final_dim]

        pred = self.linear_1(att_out.cuda())  # (bsz, max_len, |U|)
        mask = self.get_previous_user_mask(input.cuda(), self.user_size)
        output = pred.cuda() + mask.cuda()
        return output.view(-1, output.size(-1))  # (bsz*max_len, |U|)



    def get_previous_user_mask(self, seq, user_size):
        ''' Mask previous activated users.'''
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()

        masked_seq = previous_mask * seqs.data.float()
        # print(masked_seq.size())

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq


class Lstm(torch.nn.Module):
    def __init__(self,input_dim):
        super(Lstm,self).__init__()
        self.input_dim=input_dim
        self.lstm=torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=2,
            batch_first=True)
        
    
    def forward(self,x,h_state):
        r_out,h_state=self.lstm(x,h_state)

        #h_state也要作为RNN的一个输入和一个输出
        #r_out：(batch_size,time_step,HIDDEN_SIZE)
        #h_state：(batch_size,time_step,HIDDEN_SIZE)

        return r_out,h_state

    


class CEGCN_L(nn.Module):
    def __init__(self,opt):
        super(CEGCN_L, self).__init__()
        dropout=opt.dropout
        self.ntoken = opt.user_size
        self.ninp = opt.d_word_vec
        self.user_size = self.ntoken
        self.pos_dim = 8
        self.__name__ = "CEGCN"

        self.dropout = nn.Dropout(dropout)
        self.drop_timestamp = nn.Dropout(dropout)
        self.Type = opt.notes
        self.time_encoder_str=opt.time_encoder

        self.time_encoder= TimeEncoder(opt)
        if self.time_encoder_str == "Linear":
            self.time_encoder= LinearTimeEncoder(opt)
        elif self.time_encoder_str == "Slide":
            self.time_encoder= SlideTimeEncoder(opt)
        elif self.time_encoder_str == "None":
            self.time_encoder= NoneTimeEncoder(opt)

        self.user_merger=UserMerger(opt)

        self.encoder = GraphEncoder(opt,opt.dropout,self.Type)
        self.final_dim = self.ninp  # 最终的结果

        self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        self.decoder = TransformerBlock(input_size=self.final_dim + self.pos_dim+self.time_encoder.output_dim, n_heads=8)
        self.linear_1 = nn.Linear(self.final_dim +self.time_encoder.output_dim, self.ntoken)

        self.sig=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.linear_2 = nn.Linear(self.ninp, self.ninp,bias=True)
        self.linear_3= nn.Linear(self.ninp, self.ninp,bias=True)
        self.linear_4= nn.Linear(self.ninp+self.ninp, self.ninp,bias=True)
        self.linear_5= nn.Linear(self.ninp*4, self.ninp,bias=True)

        self.lstm=Lstm(input_dim=self.final_dim +self.time_encoder.output_dim)

        self.init_weights()
        print(self)

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear_1.weight)
        init.xavier_normal_(self.linear_2.weight)
        init.xavier_normal_(self.linear_3.weight)
        init.xavier_normal_(self.linear_4.weight)

    def forward(self, input, input_timestamp, input_id,train=True):
        
        input = input[:, :-1]  # [bsz,max_len]   input_id:[batch_size]

        batch_size,max_len=input.shape

        # timeencoder
        time_embedding,input_timestamp = self.time_encoder(input, input_timestamp,train)
        # item_embedding=self.item_encoder(input,input_timestamp,train)

        #encoder
        user_embedding = self.encoder(input, input_timestamp,train)
        user_att = self.user_merger(user_embedding, input_timestamp,train)
        # user_embedding = user_att
        # user_embedding = self.dropout(user_embedding)

        # score=self.sig(self.linear_2(user_embedding)+self.linear_3(user_att))
        # user_embedding=score*user_embedding+(1-score)*user_att


        user_embedding = torch.cat([user_embedding, user_att], dim=-1)
        user_embedding = self.tanh(self.linear_4(user_embedding))

        # user_embedding = torch.cat([user_embedding, user_att, user_embedding * user_att, user_embedding - user_att], dim=-1)
        # user_embedding = self.tanh(self.linear_5(user_embedding))


        user_embedding = self.dropout(user_embedding)
        if self.time_encoder_str != "None":
            user_embedding=torch.cat([user_embedding,time_embedding],dim=-1)

        # mask = (input == Constants.PAD)
        # batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        # order_embed = self.dropout(self.pos_embedding(batch_t))

        # final_input = torch.cat([user_embedding, order_embed], dim=-1).cuda()  # dynamic_node_emb
        # att_out = self.decoder(final_input.cuda(), final_input.cuda(), final_input.cuda(), mask=mask.cuda())
        hidden = None
        # user_embedding=user_embedding.view(batch_size, max_len, 72).cuda()
        att_out,hidden=self.lstm(user_embedding,hidden)

        
        att_out = self.dropout(att_out.cuda())
        # print(att_out.size())   # [bsz,max_len,pos+final_dim]

        pred = self.linear_1(att_out.cuda())  # (bsz, max_len, |U|)
        mask = self.get_previous_user_mask(input.cuda(), self.user_size)
        output = pred.cuda() + mask.cuda()
        return output.view(-1, output.size(-1))  # (bsz*max_len, |U|)



    def get_previous_user_mask(self, seq, user_size):
        ''' Mask previous activated users.'''
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()

        masked_seq = previous_mask * seqs.data.float()
        # print(masked_seq.size())

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq
