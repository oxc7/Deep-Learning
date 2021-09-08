# Overview
# 評価方法

# 評価は生成画像のテストデータに対するNLL（負の対数尤度）で行います. −∑𝐷𝑖=1𝑥𝑖log𝑥𝑖^+(1−𝑥𝑖)log(1−𝑥𝑖^)
# 定時にNLLを計算しLeader Boardを更新します。
# 締切後のNLLを最終的な評価とします
# 目標値 NLL（負の対数尤度） 235

import numpy as np import pandas as pd import torch

import torch.nn as nn import torch.optim as optim import torch.autograd as autograd import torch.nn.functional as F from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

from sklearn.utils import shuffle from sklearn.metrics import f1_score from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score

#学習データ
x_train = np.load(‘drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture07_20210527/data/x_train.npy’)

#テストデータ
x_test = np.load(‘drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture07_20210527/data/x_test.npy’)

print (x_train, x_test) class dataset(torch.utils.data.Dataset): def init(self, x_test): self.x_test = x_test.reshape(-1, 784).astype(‘float32’) / 255

def __len__(self):
    return self.x_test.shape[0]

def __getitem__(self, idx):
    return torch.tensor(self.x_test[idx], dtype=torch.float)
trainval_data = dataset(x_train) test_data = dataset(x_test)

#VAEの実装
batch_size = 100

val_size = 10000 train_size = len(trainval_data) - val_size

train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

dataloader_train = torch.utils.data.DataLoader( train_data, batch_size=batch_size, shuffle=True )

dataloader_valid = torch.utils.data.DataLoader( val_data, batch_size=batch_size, shuffle=True )

dataloader_test = torch.utils.data.DataLoader( test_data, batch_size=batch_size, shuffle=False )

import torch.nn as nn import torch.optim as optim import torch.autograd as autograd import torch.nn.functional as F

device = ‘cuda’

torch.log(0)によるnanを防ぐ
def torch_log(x): return torch.log(torch.clamp(x, min=1e-10))

class VAE(nn.Module): def init(self, z_dim): super(VAE, self).init() self.dense_enc1 = nn.Linear(2828, 200) self.dense_enc2 = nn.Linear(200, 200) self.dense_encmean = nn.Linear(200, z_dim) self.dense_encvar = nn.Linear(200, z_dim) self.dense_dec1 = nn.Linear(z_dim, 200) self.dense_dec2 = nn.Linear(200, 200) self.dense_dec3 = nn.Linear(200, 2828)

def _encoder(self, x):
    x = F.relu(self.dense_enc1(x))
    x = F.relu(self.dense_enc2(x))
    mean = self.dense_encmean(x)
    var = F.softplus(self.dense_encvar(x))
    return mean, var

def _sample_z(self, mean, var):
    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.sqrt(var) * epsilon

def _decoder(self, z):
    x = F.relu(self.dense_dec1(z))
    x = F.relu(self.dense_dec2(x))
    x = torch.sigmoid(self.dense_dec3(x))
    return x

def forward(self, x):
    mean, var = self._encoder(x)
    z = self._sample_z(mean, var)
    x = self._decoder(z)
    return x, z

def loss(self, x):
    mean, var = self._encoder(x)
    # KL lossの計算
    KL = -0.5 * torch.mean(torch.sum(1 + torch_log(var) - mean**2 - var, dim=1))

    z = self._sample_z(mean, var)
    y = self._decoder(z)

    # reconstruction lossの計算
    reconstruction = torch.mean(torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1))

    return KL, -reconstruction 
z_dim = 10 n_epochs = 300 model = VAE(z_dim).to(device) optimizer = optim.Adam(model.parameters(), lr=0.0005) for epoch in range(n_epochs): losses = [] KL_losses = [] reconstruction_losses = [] model.train() for x in dataloader_train:

    x = x.to(device)

    model.zero_grad()

    KL_loss, reconstruction_loss = model.loss(x)  # lossの各項の計算

    loss = KL_loss + reconstruction_loss  # 和を取ってlossとする

    loss.backward()
    optimizer.step()


    losses.append(loss.cpu().detach().numpy())
    KL_losses.append(KL_loss.cpu().detach().numpy())
    reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())

losses_val = []
model.eval()
for x in dataloader_valid:

    x = x.to(device)

    KL_loss, reconstruction_loss = model.loss(x)

    loss = KL_loss + reconstruction_loss

    losses_val.append(loss.cpu().detach().numpy())

print('EPOCH:%d, Train Lower Bound:%lf, (%lf, %lf), Valid Lower Bound:%lf' %
      (epoch+1, np.average(losses), np.average(KL_losses), np.average(reconstruction_losses), np.average(losses_val)))


