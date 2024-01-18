PAD = 0
UNK = 2
BOS = 3
EOS = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
time_step_split = 5
n_heads = 14
step_len = 5



import torch

device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
