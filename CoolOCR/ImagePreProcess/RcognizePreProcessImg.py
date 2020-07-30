# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   RecognizePreProcessImg
   Description : 用于识别的图像预处理基本函数
   Author:      qisen
   date:        2020/7/27 / 下午1:38
-------------------------------------------------
   Change Activity:
                   2020/7/27:
-------------------------------------------------
"""
import numpy as np

from ProjectException.DataException import DataShapeSizeError
from .PublicImgProcessFunction import *


# ================================== pydoc ==================================
# 用于识别的基础预处理图像函数
# ================================ end pydoc ================================

def imgRecognizeBinaryProcess(img, disablePixelRatio, colorFloor=256):
    """
    -------------------------------------------------------------------------------------------
    This function is used to :用于识别的图像二值化预处理
    -------------------------------------------------------------------------------------------
    Params
        :param img: 需要进行二值化变化的待识别图像
        :param disablePixelRatio: 忽略处理的像素比例
        :param colorFloor: 默认的图像色阶为256
    Return
    -------------------------------------------------------------------------------------------
    """
    inputImgDim = len(img.shape)
    if inputImgDim < 2 or inputImgDim > 4:
        exceptionMsg = ">>>ERR: 输入数据非图像，图像矩阵维度要求为3."
        imageDim = 3
        DataShapeSizeError(exceptionMsg=exceptionMsg).exceptionProcess(imageDim, inputImgDim)

    if 2 < inputImgDim <= 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 光流自适应平衡
    tileGradSize = (8, 8)
    img = opticalFlowBalance(img, tileGradSize=tileGradSize)

    # 改造分水岭方法
    img = deviationWatershed(img, disablePixelRatio, colorFloor)

    # 去除噪点
    img = cv2.medianBlur(img, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img


def cropCharacterImage(img):
    """
    -------------------------------------------------------------------------------------------
    This function is used to : 提取包含字符的最小图片区域
    -------------------------------------------------------------------------------------------
    Params
        :param img: 待抠取ROI的图像
    Return
    -------------------------------------------------------------------------------------------
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = imgRecognizeBinaryProcess(img, disablePixelRatio=0.01, colorFloor=256)

    high, width = img.shape[0], img.shape[1]

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
    x, y, w, h = cv2.boundingRect(cnt)

    x_min = max(x - 2, 0)
    y_min = max(y - 2, 0)
    x_max = min(x + w + 2, width)
    y_max = min(y + h + 2, high)

    img = img[y_min: y_max, x_min: x_max]
    # update 200407
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

    return img
