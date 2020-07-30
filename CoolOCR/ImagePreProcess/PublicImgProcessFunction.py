# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   PublicImgProcessFunction
   Description : 提供图像预处理的公用函数接口
   Author:      qisen
   date:        2020/7/27 / 下午1:57
-------------------------------------------------
   Change Activity:
                   2020/7/27:
-------------------------------------------------
"""
import cv2


def opticalFlowBalance(img, tileGradSize):
    """
    -------------------------------------------------------------------------------------------
    This function is used to : 光流自平衡函数
    -------------------------------------------------------------------------------------------
    Params
        :param img: 待处理图像数据
        :param tileGradSize: 局部光流视野大小
    Return
    -------------------------------------------------------------------------------------------
    """
    if tileGradSize is None:
        tileGradSize = (8, 8)

    clare = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGradSize)
    img = clare.apply(img)

    return img


def deviationWatershed(img, disablePixelRatio, colorFloor):
    """
    -------------------------------------------------------------------------------------------
    This function is used to :基于像素分布的分水岭提取算法
    -------------------------------------------------------------------------------------------
    Params
        :param img: 待处理图像数据
        :param disablePixelRatio: 忽略处理的像素比例
        :param colorFloor: 图像色阶
    Return
    -------------------------------------------------------------------------------------------
    """
    h, w = img.shape[0], img.shape[1]
    hist = cv2.calcHist([img], [0], None, [colorFloor], [0, colorFloor])
    numPx, px = 0, 0
    for px in range(colorFloor):
        if numPx / h / w > disablePixelRatio:
            break
        else:
            numPx += hist[px]
    _, img = cv2.threshold(img, px, colorFloor - 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return img
