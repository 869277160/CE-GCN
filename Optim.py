'''
Author: your name
Date: 2020-12-14 14:22:55
LastEditTime: 2020-12-23 09:30:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /My_papaer_base_12_9/Optim.py
'''
'''A wrapper class for optimizer '''
import numpy as np


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps,data_path):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.data_path=data_path

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self,epoch):
        ''' Learning rate scheduling per step '''
        if self.data_path.find("douban") != -1:
            if epoch < 15 :
                self.n_current_steps += 1
                new_lr = np.power(self.d_model, -0.5) * np.min([
                    np.power(self.n_current_steps, -0.5),
                    np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            elif epoch == 15:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 5.000e-04

        if self.data_path.find("twitter") != -1: 
            self.n_current_steps += 1
            new_lr = np.power(self.d_model, -0.5) * np.min([
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # if epoch < 15 :
            #     self.n_current_steps += 1
            #     new_lr = np.power(self.d_model, -0.5) * np.min([
            #         np.power(self.n_current_steps, -0.5),
            #         np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = new_lr

            # elif epoch == 15:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = 5.000e-04

        if self.data_path.find("memetracker") != -1:
            
            self.n_current_steps += 1
            new_lr = np.power(self.d_model, -0.5) * np.min([
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # if epoch < 15 :
            #     self.n_current_steps += 1
            #     new_lr = np.power(self.d_model, -0.5) * np.min([
            #         np.power(self.n_current_steps, -0.5),
            #         np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = new_lr

            # elif epoch == 15:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = 5.000e-04