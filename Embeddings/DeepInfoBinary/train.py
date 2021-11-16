import torch
from models import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator, Predictor
from datasets import SeqDataset
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


def get_logger(log_path):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1, seqL=100, encode_nc=64, feature_nc=128):
        super().__init__()
        self.global_d = GlobalDiscriminator(seqL=seqL, encode_nc=encode_nc)
        self.local_d = LocalDiscriminator(input_nc=encode_nc+feature_nc)
        self.predictor = Predictor()
        self.regression_loss = torch.nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seqL = 100

    def forward(self, y, M, M_prime, p_expr):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1)
        y_exp = y_exp.expand(-1, y.size(1), M.size(2))

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        pre_expr = self.predictor(y)
        PRE = self.regression_loss(pre_expr, p_expr)

        return LOCAL + GLOBAL, PRE


if __name__ == '__main__':
    logger = get_logger(log_path='cache/training_log/')
    parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--gpu', default=True, type=bool, help='gpu')
    args = parser.parse_args()
    seqL = 165
    encode_nc = 64
    feature_nc = 128
    input_nc = 4

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    batch_size = args.batch_size
    gpu = args.gpu

    dataset = DataLoader(dataset=SeqDataset(path='../../data/ecoli_mpra_inducible.csv', isGpu=gpu), batch_size=batch_size, shuffle=True)

    encoder = Encoder(seqL=seqL, input_nc=input_nc, encode_nc=encode_nc, feature_nc=feature_nc).to(device)
    loss_fn = DeepInfoMaxLoss(seqL=seqL, encode_nc=encode_nc, feature_nc=feature_nc).to(device)
    optim = Adam(encoder.parameters(), lr=1e-4)
    loss_optim = Adam(loss_fn.parameters(), lr=1e-4)
    epochs = 3000

    epoch_restart = 0
    root = Path(r'cache')

    if epoch_restart != 0 and root is not None:
        enc_file = root / Path('encoder' + str(epoch_restart) + '.wgt')
        loss_file = root / Path('loss' + str(epoch_restart) + '.wgt')
        encoder.load_state_dict(torch.load(str(enc_file)))
        loss_fn.load_state_dict(torch.load(str(loss_file)))

    for epoch in range(epoch_restart, epochs, 1):
        train_loss, predictor_loss = [], []
        pbar = tqdm(enumerate(dataset))
        pbar.set_description('epochs: {}'.format(epoch))
        for iter, train_loader in pbar:
            optim.zero_grad()
            loss_optim.zero_grad()
            train_input = train_loader
            y, M = encoder(train_input[0])
            # rotate images to create pairs for comparison
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss, pre_loss = loss_fn(y, M, M_prime, train_input[1])
            train_loss.append(loss.item())
            predictor_loss.append(pre_loss.item())
            (0*loss + 1*pre_loss).backward()
            optim.step()
            loss_optim.step()

        if epoch % 10 == 0:
            logger.info(str(epoch) + ' Loss: ' + str(stats.mean(train_loss)))
            logger.info(str(epoch) + ' Predictor Loss: ' + str(stats.mean(predictor_loss)))
            enc_file = root / Path('encoder' + str(epoch) + '.wgt')
            loss_file = root / Path('loss' + str(epoch) + '.wgt')
            enc_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), str(enc_file))
            torch.save(loss_fn.state_dict(), str(loss_file))
