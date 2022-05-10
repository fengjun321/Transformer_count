#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 20:03
@Author  : Xie Cheng
@File    : train_transformer.py
@Software: PyCharm
@desc: transformer训练
"""
import sys
sys.path.append("../")

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable

from myfunction import MyDataset
from Transformer.transformer import TransformerTS

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

# batch size
batch_size_train = 7
out_put_size = 3

data_bound = batch_size_train - out_put_size

# total epoch(总共训练多少轮)
total_epoch = 1000

# 1. 导入训练数据
filename = '../data/shu.csv'
dataset_train = MyDataset(filename)
train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=False, drop_last=True)

# 2. 构建模型，优化器
tf = TransformerTS(input_dim=1,
                   dec_seq_len=batch_size_train-out_put_size,
                   out_seq_len=out_put_size,
                   d_model=32,  # 编码器/解码器输入中预期特性的数量
                   nhead=8,
                   num_encoder_layers=3,
                   num_decoder_layers=3,
                   dim_feedforward=32,
                   dropout=0.1,
                   activation='relu',
                   custom_encoder=None,
                   custom_decoder=None).to(device)

optimizer = torch.optim.Adam(tf.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()  # mean square error
train_loss_list = []  # 每次epoch的loss保存起来
total_loss = 31433357277  # 网络训练过程中最大的loss


# 3. 模型训练
def train_transformer(epoch):
    global total_loss
    mode = True
    tf.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和
    for idx, (sin_input, _) in enumerate(train_loader):
        sin_input_np = sin_input.numpy()[:data_bound]  # 1D
        cos_output = sin_input[data_bound:]

        sin_input_torch = Variable(torch.from_numpy(sin_input_np[np.newaxis, :, np.newaxis]))  # 3D

        prediction = tf(sin_input_torch.to(device))  # torch.Size([batch size])
        loss = criterion(prediction, cos_output.to(device))  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        #scheduler.step()
        #print(scheduler.get_lr())

        loss_epoch += loss.item()  # 将每个batch的loss累加，直到所有数据都计算完毕
        if idx == len(train_loader) - 1:
            print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
            train_loss_list.append(loss_epoch)
            if loss_epoch < total_loss:
                total_loss = loss_epoch
                torch.save(tf, '..\\model\\tf_model2.pkl')  # save model


if __name__ == '__main__':
    # 模型训练
    print("Start Training...")
    for i in range(total_epoch):  # 模型训练1000轮
        train_transformer(i)
    print("Stop Training!")
