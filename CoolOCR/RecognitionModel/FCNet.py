# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   FCNet
   Description :
   Author:      qisen
   date:        2020/7/6 / 上午11:43
-------------------------------------------------
   Change Activity:
                   2020/7/6:
-------------------------------------------------
"""
# ================================== pydoc ==================================
# Full Connect Model to recognition the hand write number
# ================================ end pydoc ================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Model(nn.Module):
    """
    -------------------------------
    This FC_Model class is used to : 全连接神经网络模型识别手写数字字符
    -------------------------------
    """

    def __init__(self):
        """
        ---------------------------
        This function is used to : 初始化FC_Model类
        ---------------------------
        Params
            :param : NONE
        Return
        ---------------------------
        """
        super(FC_Model, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1
        )

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.8)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

        pass

    def forward(self, x):
        """
        ---------------------------
        This function is used to : 梯度前向传播
        ---------------------------
        Params
            :param x: 变量
        Return
        ---------------------------
        """
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)
        out = self.dropout1(out)
        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        out = F.log_softmax(out, dim=1)

        return out
