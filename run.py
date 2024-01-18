import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import Constants
from DataConstruct import DataConstruct
from model.CEGCN import CEGCN, CEGCN_L

from Metrics import Metrics
from Optim import ScheduledOptim


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

metric = Metrics()

root_path = './'
parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument("-data",type=str,default="twitter")
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-time_interval', type=int, default=10000)
parser.add_argument('-d_model', type=int, default=64)
# parser.add_argument('-d_inner_hid', type=int, default=64)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.15)
parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')
parser.add_argument('-log', default=None)
parser.add_argument('-save_path', default=root_path + "checkpoint/DiffusionPrediction.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', default=False)
parser.add_argument('-network', type=bool,
                    default=True)  # use social network; need features or deepwalk embeddings as initial input
parser.add_argument('-pos_emb', type=bool, default=True)
parser.add_argument('-warmup', type=int, default=10)  # warmup epochs
parser.add_argument('-notes', default="social+diffusion+item")
parser.add_argument('-time_encoder', default="")

opt = parser.parse_args()
opt.d_word_vec = opt.d_model
opt.save_path = root_path + f"checkpoint/CEGCN_{opt.data}_{int(time.time())}_{opt.notes}.pt"
# opt.data = "twittE"
# opt.data="twitter"
# opt.notes = "diag1_with_tanh_no_attwithorg_sig"
# opt.notes="linear_merger"
# opt.notes = "Dy_merger"
print(opt)

import wandb
# # torch.cuda.empty_cache()
# # if torch.cuda: torch.cuda.set_device(Constants.device)
# wandb.init(project="CEGCN_test",config=opt)
# wandb.run.name = f"CEGCN_test_{opt.data}_{int(time.time())}_{opt.notes}"

# wandb.run.save()



class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_performance(crit, pred, gold):
    ''' Apply label smoothing 
    if needed '''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    # print ("get performance, ", gold.data, pred.data)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

    true_set = set()
    for items in gold.cpu().numpy().tolist():
        true_set.add(items)
    pre_set = set()
    for item in pred.cpu().numpy().tolist():
        if item in true_set:
            pre_set.add(item)

    # if len(pre_set) / len(true_set) > 0.3 :
    #     print(gold.cpu().numpy().tolist())
    #     print(pred.cpu().numpy().tolist())

    return loss, n_correct, len(pre_set), len(true_set)



def train_epoch(model, training_data, loss_func, optimizer,epoch):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    total_same_user = 0.0
    n_total_uniq_user = 0.0
    batch_num = 0.0

    for i, batch in enumerate(
            training_data):  # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        tgt, tgt_timestamp, tgt_id = batch
        tgt.cuda()
        tgt_timestamp.cuda()

        start_time = time.time()
        import numpy as np
        np.set_printoptions(threshold=np.inf)
        gold = tgt[:, 1:].cuda()

        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        optimizer.zero_grad()
        pred = model(tgt, tgt_timestamp, tgt_id,train=True)
        # backward
        loss, n_correct, same_user, input_users = get_performance(loss_func, pred, gold)
        loss.backward(retain_graph=True)

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate(epoch)

        # note keeping
        n_total_correct += n_correct
        total_loss = total_loss + loss.item()

        total_same_user += same_user
        n_total_uniq_user += input_users

        print("Training batch ", i, " loss: ", loss.item(), " acc:", (n_correct.item() / len(pred)),
              f"\t\toutput_users:{(same_user)}/{(input_users)}={same_user / input_users}", )
        
        # wandb.log({
        # "step_loss": loss.item()})
        # print ("A Batch Time: ", str(time.time()-start_time))

    return total_loss / n_total_words, n_total_correct / n_total_words, total_same_user / n_total_uniq_user


def test_epoch(model, validation_data, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    for i, batch in enumerate(
            validation_data):  # tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
        print("Validation batch ", i)
        # prepare data
        # print(batch)

        tgt, tgt_timestamp, tgt_id = batch
        tgt.cuda()
        tgt_timestamp.cuda()

        y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

        pred = model(tgt, tgt_timestamp, tgt_id,train=False)

        y_pred = pred.detach().cpu().numpy()

        scores_batch, scores_len,MRR = metric.compute_metric(y_pred, y_gold, k_list)
        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words


    return scores,MRR


def train_model(data_path):
    # ========= Preparing DataLoader =========#

    train_data = DataConstruct(data_path, data=0, load_dict=False, batch_size=opt.batch_size, cuda=False)
    valid_data = DataConstruct(data_path, data=1, batch_size=opt.batch_size, cuda=False)  # torch.cuda.is_available()
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=False)

    opt.user_size = train_data.user_size

    # ========= Preparing Model =========#
    opt.data_path = data_path
    model = CEGCN(opt)

    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps,data_path)

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerAdam, 'max',factor=0.4, patience=3, verbose=True)


    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    validation_history = 0.0
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')
        # wandb.log({
        # "Epoch": epoch_i,
        # })
        start = time.time()
        train_loss, train_accu, train_pred = train_epoch(model, train_data, loss_func, optimizer,epoch_i)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, predected:{pred:3.3f} %' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, pred=100 * train_pred,
            elapse=(time.time() - start) / 60))
        
        # wandb.log({
        # "train_loss": train_loss,
            # })

        if epoch_i >= 0:
            start = time.time()
            scores,MRR = test_epoch(model, valid_data)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
                # wandb.log({f"Validation_{metric}":scores[metric]})
            print("Validation use time: ", (time.time() - start) / 60, "min")
            print(f"MRR: {MRR}")

               
                    
            print('  - (Test) ')
            scores,MRR = test_epoch(model, test_data)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
                # wandb.log({f"Test_{metric}":scores[metric]})
            if validation_history <= scores["hits@100"]:
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@100"], epoch_i))
                validation_history = scores["hits@100"]
                print("Save best model!!!")
                # torch.save(model.state_dict(), opt.save_path)
                print(f"MRR: {MRR}")
            
        scheduler.step(validation_history)

def test_model(data_path):
    train_data = DataConstruct(data_path, data=0, load_dict=False, batch_size=opt.batch_size, cuda=False)
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=torch.cuda.is_available())
    opt.user_size = train_data.user_size

    model = CEGCN(opt)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()

    scores,MRR= test_epoch(model, test_data)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print(f"MRR: {MRR}")

if __name__ == "__main__":
    data_path = "./data/" + opt.data
    sys.stdout = Logger(f"./log/logfile_{opt.data}_{opt.notes}__{int(time.time())}.txt")

    train_model(data_path)
    # test_model(data_path)


#   CUDA_VISIBLE_DEVICES=0  python run.py -data="twitter" &
#   CUDA_VISIBLE_DEVICES=1  python run.py -data="douban" &
#   CUDA_VISIBLE_DEVICES=2  python run.py -data="memetracker" &

# diag0 表示使用对角阵
# diag1 表示不使用用对角

# 目前工作方向
#   1.尝试加入物品之间的关系，目前测试基于杰卡德系数的关系
#   2.尝试直接加入时间关系,测试结果表明，对于meme有所提升
#   3.尝试加入历史用户影响力,目前没有思路
#   4.从原始论文中加入其他特征进去
#   5。在meme中加入一些基于共现的社交图，已经加入，结果说明对于其中有所提升
#   6.对物品进行聚类，从而将更多的用户联系到同一个物品上

# 如果要在同一个py文件内记录多个run，可以参考：

# import wandb
# for x in range(10):
#     wandb.init(reinit=True)
#     for y in range (100):
#         wandb.log({"metric": x+y})
#     wandb.join()