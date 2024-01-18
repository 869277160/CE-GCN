'''
Author: your name
Date: 2020-12-16 14:00:01
LastEditTime: 2020-12-16 21:18:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /My_papaer_base_12_9/Metrics.py
'''
import numpy as np


class Metrics(object):

    def __init__(self):
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):
        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # if not actual:
        # 	return 0.0
        return score / min(len(actual), k)

    def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
        '''
            y_true: (#samples, )
            y_pred: (#samples, #users)
        '''
        MRR=0
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        # y_prob=self.resort(y_prob,y_true)
        
        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                # p_sort = self.resort(p_,y_)
                p_sort=p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])

                # res=np.where(list_numpy == y_)
                tops = p_sort[-40000:][::-1]
                MRR += 1 / (tops[y_] + 1)

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len,MRR/(scores_len)

    # def resort(self,y_prob,y_true):
    #     res=y_prob
    #     predicted_users=set()
    #     for i in range(len(y_true)):
    #         p_ = y_prob[i]
    #         y_ = y_true[i]
    #
    #         p_sort=p_.argsort()
    #         top_1 = p_sort[-1:][::-1][0]
    #         # k=1
    #         # top_k = p_sort[-k:][::-1]
    #
    #
    #         if (y_ != 1 or y_ != 0) and top_1 == y_:
    #
    #             if len(predicted_users) != 0:
    #                 for user_idx in predicted_users:
    #                     res[i][user_idx]=y_prob[i][user_idx]-10
    #
    #             predicted_users.add(top_1)
    #
    #
    #
    #         if y_ == self.PAD or y_ == 1:
    #             predicted_users.clear()
    #             # print(len(predicted_users))
    #
    #     return res
