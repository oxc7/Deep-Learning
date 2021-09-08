# 課題. RNNを用いてIMDbのsentiment analysisを実装してみましょう。 目標値: F値 = 0.85
# 評価方法:
# 予測ラベルの（t_testに対する）F値で評価します。
# 定時に評価しLeader Boardを更新します。
# 締切後のF値でLeader Boardを更新します。これを最終的な評価とします。
# 目標値 F値：0.85

import numpy as np import torch import torchtext.legacy from torchtext.legacy import data from torchtext.legacy import datasets from sklearn.utils import shuffle from sklearn.metrics import f1_score from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score import pandas as pd import string import re

rng = np.random.RandomState(1234) random_state = 42

def preprocessing_text(text):

  text = re.sub('<br />', '', text)　　# 改行コードを消去


for p in string.punctuation:　　# カンマ、ピリオド以外の記号をスペースに置換
    if (p == ".") or (p == ","):
        continue
    else:
        text = text.replace(p, " ")


text = text.replace(".", " . ")　　 # ピリオドなどの前後にはスペースを入れておく
text = text.replace(",", " , ")
return text
#分かち書き（今回はデータが英語で、簡易的にスペースで区切る）
def tokenizer_punctuation(text): return text.strip().split()

#前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text): text = preprocessing_text(text) ret = tokenizer_punctuation(text) return ret

#文章とラベルの両方に用意します
max_length = 256 TEXT = data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="", eos_token="") LABEL = data.Field(sequential=False, use_vocab=True)

#引数の意味は次の通り init_token：全部の文章で、文頭に入れておく単語 eos_token：全部の文章で、文末に入れておく単語

#データセットの作成
train_data, test_data = datasets.IMDB.splits(text_field=TEXT,label_field=LABEL)

for i in range(len(test_data)): if i % 2 == 0: test_data[i].label = “pos” else: test_data[i].label = “neg”

train_data, valid_data = train_data.split(0.8)

#実装
import torch.nn as nn import torch.optim as optim import torch.autograd as autograd import torch.nn.functional as F

word_num = 10000 TEXT.build_vocab(train_data, max_size=word_num) LABEL.build_vocab(train_data)

batch_size = 100

train_dl = torchtext.legacy.data.Iterator(train_data, batch_size=batch_size,　train=True, sort=True)

valid_dl = torchtext.legacy.data.Iterator(valid_data, batch_size=batch_size,　train=False, sort=False)

test_dl = torchtext.legacy.data.Iterator(test_data, batch_size=batch_size,　train=False, sort=False)

def torch_log(x): return torch.log(torch.clamp(x, min=1e-10))

class Embedding(nn.Module):

def __init__(self, emb_dim, vocab_size):
    super().__init__()
    self.embedding_matrix = nn.Parameter(torch.rand((vocab_size, emb_dim),　dtype=torch.float))

def forward(self, x):
    return F.embedding(x, self.embedding_matrix)
class RNN(nn.Module): def init(self, in_dim, hid_dim): super().init() self.hid_dim = hid_dim glorot = 6/(in_dim + hid_dim*2) self.W = nn.Parameter(torch.tensor(rng.uniform( low=-np.sqrt(glorot), high=np.sqrt(glorot), size=(in_dim + hid_dim, hid_dim) ).astype(‘float32’))) self.b = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype(‘float32’)))

def function(self, h, x):
    return torch.tanh(torch.matmul(torch.cat([h, x], dim=1), self.W) + self.b)

def forward(self, x, len_seq_max=0, init_state=None):
    x = x.transpose(0, 1)  # 系列のバッチ処理のため、次元の順番を「系列、バッチ」の順に入れ替える
    state = init_state
    
    if init_state is None:  # 初期値を設定しない場合は0で初期化する
        state = torch.zeros((x[0].size()[0], self.hid_dim)).to(x.device)

    size = list(state.unsqueeze(0).size())
    size[0] = 0
    output = torch.empty(size, dtype=torch.float).to(x.device)  # 一旦空テンソルを定義して順次出力を追加する

    if len_seq_max == 0:
        len_seq_max = x.size(0)
    for i in range(len_seq_max):
        state = self.function(state, x[i])
        output = torch.cat([output, state.unsqueeze(0)])  # 出力系列の追加
    return output
class LSTM(nn.Module): def init(self, in_dim, hid_dim): super().init() self.hid_dim = hid_dim glorot = 6/(in_dim + hid_dim*2)

    self.W_i = nn.Parameter(torch.tensor(rng.uniform(
                    low=-np.sqrt(glorot),
                    high=np.sqrt(glorot),
                    size=(in_dim + hid_dim, hid_dim)
                ).astype('float32')))
    self.b_i = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

    self.W_f = nn.Parameter(torch.tensor(rng.uniform(
                    low=-np.sqrt(glorot),
                    high=np.sqrt(glorot),
                    size=(in_dim + hid_dim, hid_dim)
                ).astype('float32')))
    self.b_f = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

    self.W_o = nn.Parameter(torch.tensor(rng.uniform(
                    low=-np.sqrt(glorot),
                    high=np.sqrt(glorot),
                    size=(in_dim + hid_dim, hid_dim)
                ).astype('float32')))
    self.b_o = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

    self.W_c = nn.Parameter(torch.tensor(rng.uniform(
                    low=-np.sqrt(glorot),
                    high=np.sqrt(glorot),
                    size=(in_dim + hid_dim, hid_dim)
                ).astype('float32')))
    self.b_c = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

def function(self, state_c, state_h, x):
    i = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim=1), self.W_i) + self.b_i)
    f = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim=1), self.W_f) + self.b_f)
    o = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim=1), self.W_o) + self.b_o)
    c = f*state_c + i*torch.tanh(torch.matmul(torch.cat([state_h, x], dim=1), self.W_c) + self.b_c)
    h = o*torch.tanh(c)
    return c, h

def forward(self, x, len_seq_max=0, init_state_c=None, init_state_h=None):
    x = x.transpose(0, 1)  # 系列のバッチ処理のため、次元の順番を「系列、バッチ」の順に入れ替える
    state_c = init_state_c
    state_h = init_state_h
    if init_state_c is None:  # 初期値を設定しない場合は0で初期化する
        state_c = torch.zeros((x[0].size()[0], self.hid_dim)).to(x.device)
    if init_state_h is None:  # 初期値を設定しない場合は0で初期化する
        state_h = torch.zeros((x[0].size()[0], self.hid_dim)).to(x.device)

    size = list(state_h.unsqueeze(0).size())
    size[0] = 0
    output = torch.empty(size, dtype=torch.float).to(x.device)  # 一旦空テンソルを定義して順次出力を追加する
    
    if len_seq_max == 0:
        len_seq_max = x.size(0)
    for i in range(len_seq_max):
        state_c, state_h = self.function(state_c, state_h, x[i])
        output = torch.cat([output, state_h.unsqueeze(0)])  # 出力系列の追加
    return output
class SequenceTaggingNet(nn.Module):

def __init__(self, word_num, emb_dim, hid_dim):
    super().__init__()
    self.Emb = Embedding(emb_dim, word_num)
    self.LSTM = LSTM(emb_dim, hid_dim)
    self.Linear = nn.Linear(hid_dim, 1)

def forward(self, x, len_seq_max=0, len_seq=None, init_state=None):
    h = self.Emb(x)
    h = self.LSTM(h, len_seq_max, init_state)
    if len_seq is not None:
        # 系列が終わった時点での出力を取る必要があるので len_seq を元に集約する
        h = h[len_seq - 1, list(range(len(x))), :]
    else:
        h = h[-1]
    y = self.Linear(h)
    return y
#自分の設定
emb_dim = 100 hid_dim = 50 n_epochs = 10 lr = 0.0015 device = ‘cuda’

for epoch in range(n_epochs): losses_train = [] losses_valid = []

net.train()
n_train = 0
acc_train = 0

for mini_batch in train_dl:

    net.zero_grad()  # 勾配の初期化

    t = mini_batch.label.to(device)-1  # テンソルをGPUに移動
    x = mini_batch.text[0].to(device)
    len_seq = mini_batch.text[1].to(device)
    h = net(x, torch.max(len_seq), len_seq)
    y = torch.sigmoid(h).squeeze()
  
    loss = -torch.mean(t*torch_log(y) + (1 - t)*torch_log(1 - y))

    loss.backward()  # 誤差の逆伝播
    
    optimizer.step()  # パラメータの更新

    losses_train.append(loss.tolist())

    n_train += t.size()[0]


t_valid = []
y_pred = []
net.eval()
for mini_batch in valid_dl:

    t = mini_batch.label.to(device)-1  # テンソルをGPUに移動
    x = mini_batch.text[0].to(device)
    len_seq = mini_batch.text[1].to(device)
    h = net(x, torch.max(len_seq), len_seq)
    y = torch.sigmoid(h).squeeze()
    
    loss = -torch.mean(t*torch_log(y) + (1 - t)*torch_log(1 - y))

    pred = y.round().squeeze()  # 0.5以上の値を持つ要素を正ラベルと予測する

    t_valid.extend(t.tolist())
    y_pred.extend(pred.tolist())

    losses_valid.append(loss.tolist())

print('EPOCH: {}, Train Loss: {:.3f}, Valid Loss: {:.3f}, Validation F1: {:.3f}'.format(
    epoch,
    np.mean(losses_train),
    np.mean(losses_valid),
    f1_score(t_valid, y_pred, average='macro')
))

  
