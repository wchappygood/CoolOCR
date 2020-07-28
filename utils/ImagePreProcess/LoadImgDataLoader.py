# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   LoadImgDataLoader
   Description : 重载torch.dataset.dataLoader方法，提供加载本地数据
   Author:      qisen
   date:        2020/7/27 / 上午9:43
-------------------------------------------------
   Change Activity:
                   2020/7/27:
-------------------------------------------------
"""
# ================================== pydoc ==================================
# 重写数据加载类
# ================================ end pydoc ================================

import gzip
import os
import numpy as np

from torch.utils.data.dataset import Dataset
from ProjectException.UserException import UserIllegalFilePathError


def LoadLocalData(folder, dataFileName, dataLabelName):
    """
    -------------------------------------------------------------------------------------------
    This function is used to :加载本地数据集
    -------------------------------------------------------------------------------------------
    Params
        :param folder: 数据集文件夹路径
        :param dataFileName: 数据集文件名称
        :param dataLabelName: 数据标签名称
    Return
    -------------------------------------------------------------------------------------------
    """
    labelFilePath = os.path.join(folder, dataLabelName)
    if os.path.exists(labelFilePath) is False:
        exceptionMsg = ">>>ERR: 用户输入的路径文件不存在."
        UserIllegalFilePathError(exceptionMsg=exceptionMsg).exceptionProcess(labelFilePath)
    else:
        with gzip.open(labelFilePath) as labelFile:
            print(">>>INFO: 读取标签文件 {}...".format(dataLabelName))
            buf = labelFile.read()

            offset = 0
            header = np.frombuffer(buf, '>i', 2, offset)
            magic_number, num_labels = header
            print(">>>INFO: 魔法数：{} \n"
                  "         标签数：{}.".format(magic_number, num_labels))

            offset += header.size * header.itemsize
            y_train = np.frombuffer(buf, '>B', num_labels, offset)

    imgFilePath = os.path.join(folder, dataFileName)
    if os.path.exists(imgFilePath) is False:
        exceptionMsg = ">>>ERR: 用户输入的路径文件不存在."
        UserIllegalFilePathError(exceptionMsg=exceptionMsg).exceptionProcess(imgFilePath)
    else:
        with gzip.open(imgFilePath) as imgFile:
            print(">>>INFO: 读取图片数据 {}...".format(dataFileName))
            buf = imgFile.read()

            offset = 0
            header = np.frombuffer(buf, '>i', 4, offset)
            magic_number, num_images, num_cols, num_rows = header
            print(">>>INFO: 魔法数：{} \n"
                  "         图片数：{} \n"
                  "         图片行数：{} \n"
                  "         图片列数：{} \n".format(magic_number, num_images, num_cols, num_rows))

            offset += header.size * header.itemsize
            x_train = np.frombuffer(buf, '>B', num_images * num_rows * num_cols, offset).reshape(
                num_images, num_rows, num_cols
            )
            # 转置
            x_train = x_train.transpose(0, 2, 1)

            return x_train, y_train


class LocalDataset(Dataset):
    """
    -------------------------------------------------------------------------------------------
    This LocalDataSet class is used to : 重写torch.Dataset类，提供加载本地数据集方法
    -------------------------------------------------------------------------------------------
    """

    def __init__(self, folder, dataFileName, dataLabelName, transform=None):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :初始化本地数据集类
        -------------------------------------------------------------------------------------------
        Params
            :param folder: 数据存储文件夹
            :param dataFileName: 数据存储文件名称
            :param dataLabelName: 数据标签名
            :param transform: 数据是否需要进行转换
        Return
        -------------------------------------------------------------------------------------------
        """
        (train_set, train_labels) = LoadLocalData(folder, dataFileName, dataLabelName)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        """
        -------------------------------------------------------------------------------------------
        This function is used to : 重写
        -------------------------------------------------------------------------------------------
        Params
            :param index: 数据目录
        Return
        -------------------------------------------------------------------------------------------
        """
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :重写
        -------------------------------------------------------------------------------------------
        Params
            :param :
        Return
        -------------------------------------------------------------------------------------------
        """
        return len(self.train_set)

# End code
