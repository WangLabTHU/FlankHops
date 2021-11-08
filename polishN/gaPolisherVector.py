from SeqRegressionModel import *
from wgan_attn import *
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random
from torch.utils.data import DataLoader
import collections
import pandas as pd
from sko.GA import GA
from sko.tools import set_run_mode

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class LoadData(Dataset):

    def __init__(self, data, is_train=True, gpu_ids='0'):
        self.storage = []
        self.gpu_ids = gpu_ids
        for i in range(np.size(data, 0)):
            self.storage.append(data[i])

    def __getitem__(self, item):
        in_seq = transforms.ToTensor()(self.storage[item])
        if len(self.gpu_ids) > 0:
            return in_seq[0, :].float().cuda()
        else:
            return in_seq[0, :].float()

    def __len__(self):
        return len(self.storage)


def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if word1[i - 1] == word2[j - 1]:
                temp = 0
            else:
                temp = 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def similarity_func(word1, word2):
    res = edit_distance(word1, word2)
    maxLen = max(len(word1), len(word2))
    return 1-res*1.0/maxLen


def one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        encoded[charmap[seq[i]], i] = 1
    return encoded


def backbone_one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            encoded[charmap[seq[i]], i] = 1
    return encoded


def decode_oneHot(seq):
    keys = ['A', 'T', 'C', 'G', 'M', 'N', 'H', 'Z']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        pos = np.argmax(seq[:, i])
        dSeq += keys[pos]
    return dSeq


class Polisher:

    def __init__(self,
                 predictor_path='../Predictor/results/model/expr_deepgoplus_0.44.pth',
                 generator_path='../Generator/check_points/3UTRnet_G_9099.pth',
                 epochs=5000,
                 lr=0.01,
                 is_gpu=True,
                 seqL=100,
                 gen_num=3,
                 similarity_penalty=0.9,
                 model_iter=20099,
                 polishE=10,
                 position_sample_num=1024,
                 save_path='results/polisher_results_deepgoplus_0.44_polish_10.txt'):
        self.generator = torch.load(generator_path)
        self.predictor = torch.load(predictor_path)
        self.position_sample_num = position_sample_num
        for p in self.generator.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.epochs = epochs
        self.lr = lr
        self.is_gpu = is_gpu
        self.seqL = seqL
        self.gen_num = gen_num
        self.save_path = save_path
        self.similarity_penalty = similarity_penalty
        self.polishE = polishE
        self.seqs, self.masks, self.randns = [], [], []
        self.best_expr, self.best_seq = 0, ''
        self.seq_results, self.expr_results, self.control_results = collections.OrderedDict(), collections.OrderedDict(), collections.OrderedDict()

    def set_input(self, seqs):
        self.seqs_string = seqs
        for i in range(len(seqs)):
            seq_i = seqs[i]
            self.seq_results[seq_i], self.expr_results[seq_i] = [], []
            for j in range(self.gen_num):
                self.seq_results[seq_i].append(seq_i)
                self.expr_results[seq_i].append(0.0)
            self.seq_results[seq_i], self.expr_results[seq_i] = np.array(self.seq_results[seq_i]), np.array(self.expr_results[seq_i])
            self.seqs.append(backbone_one_hot(seqs[i]))
            self.i = 0

    def opt_func(self, p):
        p1, p2 = p[:, 0: self.polishE], p[:, self.polishE: ]
        p_reshape = np.zeros([np.size(p, 0), 4, self.seqL])
        seqs_position = np.zeros([4, self.seqL])
        for i in range(np.size(p, 0)):
            seqs_position[:, :] = self.seqs[self.i][:, :]
            seqs_position[:, np.int64(p1[i, :])] = p2[i, :].reshape([4, -1])
            p_reshape[i, :, :] = seqs_position[:, :]
        positionData = DataLoader(LoadData(data=p_reshape), batch_size=1024, shuffle=False)
        tensorSeq, pred_value = [], []
        for j, eval_data in enumerate(positionData):
            tensorSeq.append(self.generator(eval_data).detach())
        tensorSeq = torch.cat(tensorSeq, dim=0).cpu().float().numpy()
        for i in range(np.size(p, 0)):
            for j in range(self.seqL):
                maxId = np.argsort(tensorSeq[i, :, j])
                tensorSeq[i, :, j] = 0
                tensorSeq[i, maxId[-1], j] = 1
        generateData = DataLoader(LoadData(data=tensorSeq), batch_size=1024, shuffle=False)
        predictions = []
        seq_generate = []
        for j, eval_data in enumerate(generateData):
            seq_generate.append(eval_data)
            predictions.append(self.predictor(eval_data).detach())
        seq_generate = torch.cat(seq_generate, dim=0).cpu().float().numpy()
        predictions = torch.cat(predictions, dim=0).cpu().float().numpy()
        preList = np.argsort(-predictions)
        seq_max = seq_generate[preList[0]]
        expression_eval = predictions[preList[0]]

        seq_opt = decode_oneHot(np.squeeze(seq_max))
        if 'GGGGG' in seq_opt or 'CCCCC' in seq_opt: expression_eval = 0
        if expression_eval > min(self.expr_results[self.seqs_string[self.i]]):
            if seq_opt not in list(self.seq_results[self.seqs_string[self.i]]):
                self.seq_results[self.seqs_string[self.i]][-1] = seq_opt
                self.expr_results[self.seqs_string[self.i]][-1] = expression_eval
                sort_idx = np.argsort(-self.expr_results[self.seqs_string[self.i]])
                self.seq_results[self.seqs_string[self.i]] = self.seq_results[self.seqs_string[self.i]][sort_idx]
                self.expr_results[self.seqs_string[self.i]] = self.expr_results[self.seqs_string[self.i]][sort_idx]
        if expression_eval > self.best_expr:
            self.best_expr = expression_eval
            self.best_seq = seq_opt
            print('best seq:{} values:{}'.format(self.best_seq, 2**expression_eval))
        return -predictions


    def optimization(self):
        keys = ['A', 'T', 'C', 'G']
        mode = 'vectorization'
        set_run_mode(self.opt_func, mode)
        for i in range(len(self.seqs)):
            self.best_expr, self.best_seq = 0, ''
            print('Optimize seq {}'.format(i))
            seq_i = self.seqs_string[i]
            self.control_results[seq_i] = seq_i
            self.i = i
            lb, ub, precision = [], [], []
            for j in range(self.polishE + 4*self.polishE):
                if j >= 0 and j < self.polishE:
                    lb.append(0)
                    ub.append(self.seqL - 1)
                    precision.append(1)
                else:
                    lb.append(0)
                    ub.append(1)
                    precision.append(1e-7)
            ga = GA(func=self.opt_func, n_dim=self.polishE + 4*self.polishE, size_pop=12*1024, max_iter=100, prob_mut=0.005, lb=lb, ub=ub,
                    precision=precision)
            ga.run()
        with open(self.save_path, "w") as f:
            i = 0
            for seq in self.seqs_string:
                f.write('seq {} optimize results:\n'.format(i))
                control_seq = self.control_results[seq]
                seq_control_eval = transforms.ToTensor()(one_hot(control_seq)).float()
                if self.is_gpu:
                    seq_control_eval = seq_control_eval.cuda()
                expression_eval = self.predictor(seq_control_eval)
                f.write('Origin:{} predict_expression:{}\n'.format(control_seq, 2**expression_eval.item()))
                for j in range(self.gen_num):
                    f.write('Polish:{} optimize expression: {}\n'.format(self.seq_results[seq][j], 2**self.expr_results[seq][j]))
            i = i + 1
        f.close()



seq = ['ACGGCTCGGAGTTCGTCTACGGGCTGCTCTTTCTGCCCGTCTCCGCTCTGACCGCCCTCTGGGTCCGGCCCGCCGACCTGGTCACCGCTCCGATCAGCGT'
       'GGGCCGGTCATGTCCGGCGGGCCCGGGAGAGCACCGACCGCCGGGTGGTGCACGTGTACTACGAGGCGAGCGCCCGGACGGCCGCCCGCGACTACTTCAT',
       'CTCGCCGAGACGTCCGAGGAGGACGACGAGGGCTGAGCGGCGGAAGCCGATCAAGGGCGGGCCGTGAACCGGGTGCCTACCGGAACACGGCCCGCCCTTT',
       'ATTGTTTGCGGCGGAGGATTCGCCTAGTGGCCTAGGGCGCACGCTTGGAAAGCGTGTTGGGGGCAACCCCTCACGAGTTCGAATCTCGTATCCTCCGCCA',
       'CACGCAGTACACGCAGGTCCGCCCCGCGCGATTAGCTCAGCGGGAGAGCGCTTCCCTGACACGGAAGAGGTCACTGGTTCAATCCCAGTATCGCGCACCA',
       ]
op = Polisher(polishE=5, save_path='results/polisher2_results_deepgoplus_3UTR.txt')
op.set_input(seq)
op.optimization()