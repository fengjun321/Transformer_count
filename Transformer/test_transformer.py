#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 20:45
@Author  : Xie Cheng
@File    : test_transformer.py
@Software: PyCharm
@desc: transformer 测试
"""

import sys
sys.path.append("../")

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from myfunction import MyDataset

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('You are using: ' + str(device))

# batch size
batch_size_test = 7
out_put_size = 3

data_bound = batch_size_test - out_put_size

# 导入数据
filename = '../data/shu.csv'
dataset_test = MyDataset(filename)
test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=True)

criterion = nn.MSELoss()  # mean square error


# rnn 测试
def test_rnn():
    net_test = torch.load('..\\model\\tf_model2.pkl')  # load model
    test_loss = 0
    net_test.eval()
    with torch.no_grad():
        for idx, (sin_input, _) in enumerate(test_loader):
            sin_input_np = sin_input.numpy()[:data_bound]  # 1D
            cos_output = sin_input[data_bound:]
     
            sin_input_torch = torch.from_numpy(sin_input_np[np.newaxis, :, np.newaxis])  # 3D
            prediction = net_test(sin_input_torch.to(device))  # torch.Size([batch size])
            print("-------------------------------------------------")
            print("输入:", sin_input_np)
            print("预期输出:", cos_output)
            print("实际输出:", prediction)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if idx == 0:
                predict_value = prediction
                real_value = cos_output
            else:
                predict_value = torch.cat([predict_value, prediction], dim=0)
                real_value = torch.cat([real_value, cos_output], dim=0)

            loss = criterion(prediction, cos_output.to(device))
            test_loss += loss.item()

    print('Test set: Avg. loss: {:.9f}'.format(test_loss))
    return predict_value, real_value


if __name__ == '__main__':
    # 模型测试
    print("testing...")
    p_v, r_v = test_rnn()

    # 对比图
    plt.plot(p_v.cpu(), c='green')
    plt.plot(r_v.cpu(), c='orange', linestyle='--')
    plt.show()
    print("stop testing!")

