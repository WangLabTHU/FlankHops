import pandas as pd
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class SeqDataset(Dataset):

    def __init__(self, path='data.csv', isGpu=True):
        self.path = path
        seqList = list(pd.read_csv(self.path)['realB'])
        exprList = list(pd.read_csv(self.path)['expr'])
        self.pSeq = []
        self.expr = []
        self.isReal = []
        self.isGpu = isGpu

        for i in range(len(seqList)):
            self.pSeq.append(self.oneHot(seqList[i]))
            self.expr.append(exprList[i])

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        Y = self.expr[item]
        Y = transforms.ToTensor()(np.asarray([[Y]]))
        Y = torch.squeeze(Y)
        Y = Y.float()
        if self.isGpu:
            X = X.cuda()
            Y = Y.cuda()
        return X, Y

    def __len__(self):
        return len(self.pSeq)


