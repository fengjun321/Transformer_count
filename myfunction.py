#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 15:10
@Author  : Xie Cheng
@File    : myfunction.py
@Software: PyCharm
@desc: 一些自定义的函数
"""
import numpy as np
from torch.utils.data import Dataset


# 导入数据集的类
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.lines = open(csv_file).readlines()

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        cur_line = self.lines[index].split(',')

        sin_input = np.float32(cur_line[1].strip())
        cos_output = np.float32(cur_line[2].strip())

        return sin_input, cos_output

    def __len__(self):
        return len(self.lines)  # MyDataSet的行数
