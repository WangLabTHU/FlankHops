from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn


class ResBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size=13, padding=6, bias=True):
        super(ResBlock, self).__init__()
        model = [nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + 0.3*self.model(x)


class Encoder(nn.Module):
    def __init__(self, seqL=100, input_nc=4, encode_nc=64, feature_nc=128):
        super().__init__()
        self.c0 = nn.Conv1d(input_nc, 64, kernel_size=5, stride=1, padding=2)
        self.c1 = nn.Conv1d(64, feature_nc, kernel_size=5, stride=1, padding=2)
        self.res_b = [ResBlock(feature_nc, feature_nc),
                      ResBlock(feature_nc, feature_nc),
                      ResBlock(feature_nc, feature_nc),]
        self.res_b = nn.Sequential(*self.res_b)
        self.c2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.c3 = nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2)
        self.l1 = nn.Linear(512*seqL, encode_nc)

        self.b1 = nn.BatchNorm1d(feature_nc)
        self.b2 = nn.BatchNorm1d(256)
        self.b3 = nn.BatchNorm1d(512)

    def forward(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.b1(self.c1(h)))
        features = self.res_b(features)
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        encoded = self.l1(h.view(x.shape[0], -1))
        return encoded, features


class GlobalDiscriminator(nn.Module):
    def __init__(self, seqL=100, encode_nc=64):
        super().__init__()
        self.res_b = [ResBlock(128, 128),
                      ResBlock(128, 128),
                      ResBlock(128, 128)]
        self.res_b = nn.Sequential(*self.res_b)
        self.c0 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.c1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.l0 = nn.Linear(32 * seqL + encode_nc, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(self.res_b(M)))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, input_nc=192):
        super().__init__()
        self.c0 = nn.Conv1d(input_nc, 512, kernel_size=1)
        self.res_b = [ResBlock(512, 512),
                      ResBlock(512, 512)]
        self.res_b = nn.Sequential(*self.res_b)
        self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        self.c2 = nn.Conv1d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = self.res_b(h)
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 1)

    def forward(self, x):
        encoded = x
        pre = self.l1(encoded)
        return pre


class DeepInfoAsLatent(nn.Module):
    def __init__(self, run, epoch):
        super().__init__()
        model_path = Path(r'c:/data/deepinfomax/models') / Path(str(run)) / Path('encoder' + str(epoch) + '.wgt')
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(str(model_path)))
        self.classifier = Classifier()

    def forward(self, x):
        z, features = self.encoder(x)
        z = z.detach()
        return self.classifier((z, features))