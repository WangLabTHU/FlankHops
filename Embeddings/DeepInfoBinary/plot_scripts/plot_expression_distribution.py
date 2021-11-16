import torch
from models import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats
import argparse
import logging
import time
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import umap

FIG_HEIGHT = 5
FIG_WIDTH = 5


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


def oneHot(sequence):
    oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    oh = np.zeros([4, len(sequence)])
    for i in range(len(sequence)):
        oh[oh_dict[sequence[i]], i] = 1
    return oh


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
        if self.isGpu:
            X = X.cuda()
        Y = self.expr[item]
        return X, Y

    def __len__(self):
        return len(self.pSeq)


def main():
    seqL = 165
    encode_nc = 64
    feature_nc = 128
    input_nc = 4
    batch_size=32
    dataset = DataLoader(dataset=SeqDataset(path='../../../data/ecoli_mpra_inducible.csv', isGpu=True),
                         batch_size=batch_size, shuffle=True)

    control = [
               'AGCCCATCTGTGGTCAACGCAAATTAAAGCGTGCGACACACCCGAATTTTTCTAGGAAATTTGGGCAGAGTTGCTTGTATCGTAAATATCTGTATGTCTGCGAGGGCATACGTATAACTGAATATCGAGAAATTATCCATCCACTATTAGAAACCATATATTTAC',
               'ATTTTTCGAAAGCAGAGAATTTTTCTCTGCTTTTTGTTATAATAGTAAGGTTTCTTGAAAAAGTAGATATATTTTAGGCAGATTGGTTTACAACTATAATAAGCTGTACTATAATTCAAATAGATATAAATCGGAATCATTCTGAATTAAAATAGGTGAGATGCT',
               'CACACAGCGTTTAGCCTGAGTTCAATGAATAAGATACCAAGCCTTATAAAGGCAATAGGTATTATCTTTACCAACAAATATTTAGATAACGGGTAGGTCCTGCCTATCCGTTTTTGCTAAATAGTGCTATAATGAAAGAGTAAACTAAATAGATAGGAGAAACAC']
    polished = ['TAAATATTAAAATTAAAAAACTATACGGGCTGTAGGGGCTAAAATTTCTGAGTAAATTTCTAAATCGTATTATTGACATCTTTCGTTTTAGTGGTTATAATCTTCCGTAATTGTGAGCGGATAACAAGGAAGATTACTTTGTGGATTTTTTAAAGGAGAGACTTG']
    for i in range(len(control)):
        control[i] = oneHot(control[i])
    for i in range(len(polished)):
        polished[i] = oneHot(polished[i])
    control = torch.squeeze(transforms.ToTensor()(np.asarray(control))).float().cuda().reshape([-1, input_nc, seqL])
    polished = torch.squeeze(transforms.ToTensor()(np.asarray(polished))).float().cuda().reshape([-1, input_nc, seqL])

    root = Path(r'../cache')
    load_epoch = 270
    model_path = root / Path('encoder' + str(load_epoch) + '.wgt')
    encoder = Encoder(seqL=seqL, input_nc=input_nc, encode_nc=encode_nc, feature_nc=feature_nc).cuda()
    encoder.load_state_dict(torch.load(str(model_path)))
    encoder = encoder.eval()
    pbar = tqdm(enumerate(dataset))
    encode_results, expr_value = [], []
    for iter, encode_loader in pbar:
        X, Y = encode_loader
        encodings, _ = encoder(X)
        encode_results.append(encodings.detach())
        expr_value += list(Y.cpu().float().numpy())
    encode_results = torch.cat(encode_results, dim=0)
    encode_results = encode_results.cpu().float().numpy()

    control_encodings, _ = encoder(control)
    case_encodings, _ = encoder(polished)
    control_encodings = control_encodings.detach().cpu().float().numpy()
    case_encodings = case_encodings.detach().cpu().float().numpy()
    encode_results_all = np.vstack((encode_results, control_encodings))
    encode_results_all = np.vstack((encode_results_all, case_encodings))

    umap1 = TSNE(random_state=12)
    umap_emb = umap1.fit_transform(encode_results_all)

    umap_emb_0 = umap_emb[0 : np.size(encode_results, 0), :]
    umap_emb_1 = umap_emb[np.size(encode_results, 0): np.size(encode_results, 0) + np.size(control_encodings, 0), :]
    umap_emb_2 = umap_emb[np.size(encode_results, 0) + np.size(control_encodings, 0): np.size(encode_results, 0) + np.size(control_encodings, 0) + np.size(case_encodings, 0), :]
    MEDIUM_SIZE = 8
    SMALLER_SIZE = 6
    plt.rc('font', size=SMALLER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('xtick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc('figure', titlesize=MEDIUM_SIZE)
    plt.rc('font', family='Helvetica', size=SMALLER_SIZE)
    plt.rc('legend', fontsize=0.7 * SMALLER_SIZE)

    # plt.clf()
    fig, ax = plt.subplots(figsize=(1.2 * FIG_WIDTH, FIG_HEIGHT))
    csfont = {'family': 'Helvetica'}

    realP = plt.scatter(umap_emb_0[:, 0], umap_emb_0[:, 1], c=expr_value, cmap='bwr', vmin=-6, vmax=9, alpha=0.5, s=1,
                      label='natural promoter')
    plt.scatter(umap_emb_1[:, 0], umap_emb_1[:, 1], c='k', marker='o', s=3, label='control')
    plt.scatter(umap_emb_2[:, 0], umap_emb_2[:, 1], c='g', marker='o', s=3, label='case')
    cbar = plt.colorbar(realP)
    cbar.set_label("expression", fontdict=csfont)
    plt.legend(loc='upper left', frameon=False)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    # plt.axis('off')
    # for axis in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[axis].set_linewidth(2)
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.legend(['auroc >= {}'.format(up_T), 'auroc <= {}'.format(down_T)])
    # plt.title('Text Embeddings Tsne', fontdict=csfont)
    fig.tight_layout()
    #plt.show()
    plt.savefig('figures/expression_distribution.pdf')
    debug = 0

if __name__ == '__main__':
    main()