import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from metric_dataset import SeqDataset
from SeqRegressionModel import Seq2Scalar
from matplotlib import pyplot as plt
import numpy as np
import collections
import pandas as pd
from utils import EarlyStopping
from tqdm import tqdm
import logging
import time


def get_logger(log_path=''):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = log_path
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class Seq2ScalarTraining:
    def __init__(self,
                 logger,
                 train_path='',
                 val_path='',
                 fold_i=0,
                 model_name=''):
        self.batch_size = 128
        self.lr = 0.001
        self.lr_expr = 0.005
        self.gpu = True
        self.patience = 10
        self.epoch = 15
        self.seqL = 165
        self.k_fold = 3
        self.fold_i = fold_i
        self.mode = 'denselstm'
        self.model_name = model_name
        self.logger = logger
        self.dataset_train = DataLoader(dataset=SeqDataset(path=train_path, isTrain=True, isGpu=self.gpu), batch_size=self.batch_size, shuffle=True)
        self.dataset_val = DataLoader(dataset=SeqDataset(path=val_path, isTrain=True, isGpu=self.gpu),
                                        batch_size=self.batch_size, shuffle=True)
        self.model_ratio = Seq2Scalar(input_nc=4, seqL=self.seqL, mode=self.mode)
        self.save_path = 'results/model/'
        if self.gpu:
            self.model_ratio=self.model_ratio.cuda()
        self.loss_y = torch.nn.MSELoss()
        if self.mode == 'deepinfomax':
            self.optimizer_ratio = torch.optim.Adam(self.model_ratio.fc.parameters(), lr=self.lr_expr)
        else:
            self.optimizer_ratio = torch.optim.Adam(self.model_ratio.parameters(), lr=self.lr_expr)

    def training(self):
        logger = self.logger

        for epoch_i in range(self.epoch):
            train_loss_y = 0
            train_num_y = 0
            test_loss_y = 0
            test_num = 0
            self.model_ratio.train()
            print('Training iters')
            for trainLoader in tqdm(self.dataset_train):
                train_data, train_y = trainLoader['x'], trainLoader['z']
                predict = self.model_ratio(train_data)
                predict_y = torch.squeeze(predict)
                loss_y = self.loss_y(predict_y, train_y)
                self.optimizer_ratio.zero_grad()
                loss_y.backward()
                self.optimizer_ratio.step()
                train_loss_y += loss_y
                train_num_y = train_num_y + 1
            test_predict_expr = []
            test_real_expr = []
            self.model_ratio.eval()
            print('Test iters')

            with torch.no_grad():
                for testLoader in tqdm(self.dataset_val):
                    test_data, test_y = testLoader['x'], testLoader['z']
                    predict_y = self.model_ratio(test_data)
                    predict_y = predict_y.detach()
                    predict_y2 = predict_y
                    predict_y = predict_y.cpu().float().numpy()
                    predict_y = predict_y[:]
                    real_y = test_y.cpu().float().numpy()
                    for i in range(np.size(real_y)):
                        test_real_expr.append(real_y[i])
                        test_predict_expr.append(predict_y[i])
                    test_loss_y += self.loss_y(predict_y2, test_y)
                    test_num = test_num + 1
            coefs = np.corrcoef(test_real_expr, test_predict_expr)
            coefs = coefs[0, 1]
            logger.info('epoch: {} fold: {} test_coefs: {}'.format(epoch_i, self.fold_i, coefs))
            torch.save(self.model_ratio, self.model_name + '.pth')


def main():
    k_fold = 3
    logger_path = 'results/training_log/ecoli_mpra_nbps_prediction.log'
    logger = get_logger(log_path=logger_path)
    for i in range(k_fold):
        logger.info('Optimization Predictor')
        train_path = '../data/ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_train_fold_{}.csv'.format(i)
        val_path = '../data/ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_val_fold_{}.csv'.format(i)
        model_name = 'results/model/ecoli_mpra_nbps_prediction_opt_{}'.format(i)
        analysis = Seq2ScalarTraining(logger, fold_i=i, train_path=train_path, val_path=val_path, model_name=model_name)
        analysis.training()

        logger.info('Metrics Predictor')
        train_path = '../data/ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_val_fold_{}.csv'.format(i)
        val_path = '../data/ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_train_fold_{}.csv'.format(i)
        model_name = 'results/model/ecoli_mpra_nbps_prediction_metric_{}'.format(i)
        analysis = Seq2ScalarTraining(logger, fold_i=i, train_path=train_path, val_path=val_path, model_name=model_name)
        analysis.training()

if __name__ == '__main__':
    main()
