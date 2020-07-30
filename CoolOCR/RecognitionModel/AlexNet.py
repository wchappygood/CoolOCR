# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   AlexNet
   Description :
   Author:      qisen
   date:        2020/7/6 / 上午11:36
-------------------------------------------------
   Change Activity:
                   2020/7/6:
-------------------------------------------------
"""
# ================================== pydoc ==================================
# define the AlexNet Model to recognition the alphabet
# ================================ end pydoc ================================

import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    -------------------------------
    This AlexNet class is used to : 改造Alex Net用来识别英文字符
    -------------------------------
    """

    def __init__(self, imgChannel):
        """
        ---------------------------
        This function is used to : 初始化AlexNet识别方法
        ---------------------------
        Params
            :param imgChannel: 图像通道数, 默认为3通道输入
        Return
        ---------------------------
        """
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=imgChannel, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(size=5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(size=5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 27)

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
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.view(-1, 256 * 4 * 4)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out
