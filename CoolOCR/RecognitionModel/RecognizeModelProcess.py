# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   RecognizeModelProcess
   Description : 提供识别模型的训练，测试和预测函数
   Author:      qisen
   date:        2020/7/27 / 下午3:43
-------------------------------------------------
   Change Activity:
                   2020/7/27:
-------------------------------------------------
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from .AlexNet import AlexNet
from .FCNet import FC_Model
from utils.ImagePreProcess.LoadImgDataLoader import LocalDataset
from utils.ImagePreProcess.RcognizePreProcessImg import cropCharacterImage
from ProjectException.UserException import UserIllegalInputParamError


def predictImg(img, device, modelType):
    """
    -------------------------------------------------------------------------------------------
    This function is used to : 预测图片中的字符内容，支持手写数字和手写字母
    -------------------------------------------------------------------------------------------
    Params
        :param img: 待预测的图片
        :param device: cpu 或 gpu 设备编号
        :param modelType: 模型类型，支持预测数字或预测字母
    Return
    -------------------------------------------------------------------------------------------
    """
    if modelType not in {"letters", "digits"}:
        exceptionMsg = ">>>ERR: 错误预测类型.模型仅接受[letters, digits]两种预测参数."
        UserIllegalInputParamError(exceptionMsg=exceptionMsg).exceptionProcess(modelType)
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # 图像数据序列化
        img = cropCharacterImage(img)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)

        lettersModelData = "/home/qisen/git_work/CoolOCR/utils/RecognitionModel/AlphabetRecognizePerTrainModel" \
                           "/AlexNet_letter.pth"
        digitsModelData = "/home/qisen/git_work/CoolOCR/utils/RecognitionModel/AlphabetRecognizePerTrainModel" \
                          "/FC_Model.pth"

        # 加载预训练模型数据
        if modelType is "letters":
            model = torch.load(lettersModelData)
        else:
            model = torch.load(digitsModelData)
        model.eval()
        model.to(device)

        with torch.no_grad():
            imageData = img.to(device)
            out = model(imageData)
            if modelType is "letters":
                # 字母只预测A B C D E的概率，可修改
                _, pred = torch.max(out.data[:, :6], 1)
            else:
                _, pred = torch.max(out.data, 1)
            ans = pred.item()

            if modelType is "letters":
                ans += ord('A') - 1
            else:
                ans += ord('0')
            return ans


def train(args, model, device, train_loader, optimizer, epoch):
    """
    -------------------------------------------------------------------------------------------
    This function is used to : 模型训练函数
    -------------------------------------------------------------------------------------------
    Params
        :param args: 模型超参数
        :param model: 模型网络
        :param device: cpu 或 GPU 设备编号
        :param train_loader: 训练数据加载器
        :param optimizer: 模型优化算法
        :param epoch: 迭代次数
    Return
    -------------------------------------------------------------------------------------------
    """
    model.train()
    # 定义交叉熵损失函数
    loss_F = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_F(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def test(model, device, test_loader):
    """
    -------------------------------------------------------------------------------------------
    This function is used to : 模型测试函数
    -------------------------------------------------------------------------------------------
    Params
        :param model: 预训练模型数据
        :param device: cpu 或 GPU 设备编号
        :param test_loader: 测试数据加载器
    Return
    -------------------------------------------------------------------------------------------
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        print("-" * 20)
        print(">>>INFO: Accuracy: ({:.2f})%".format(correct / total * 100))


def learningModel(modelType):
    """
    ---------------------------
    This function is used to : learning model for pre-training AlexNet and FCNet_model
    ---------------------------
    Params
        :param modelType: recognition type
    Return
        pre-training model parameters
    ---------------------------
    """
    # global data_train, data_test
    parser = argparse.ArgumentParser(description='PyTorch E-MNIST AlexNetNN')

    # parser.add_argument("--stage", type=str, default='train', help="is train or predict")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument("--img_size", type=tuple, default=(28, 28), help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    # parser.add_argument("--predictImg", type=str, default='', help="image need to be predicted")

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    if modelType is "letters":
        data_train = LocalDataset("../../../Data/RecognitionData/EMNIST_DATA/",
                                  "emnist-letters-train-images-idx3-ubyte.gz",
                                  "emnist-letters-train-labels-idx1-ubyte.gz",
                                  transform=transform_train)
        data_test = LocalDataset("../../../Data/RecognitionData/EMNIST_DATA/",
                                 "emnist-letters-test-images-idx3-ubyte.gz",
                                 "emnist-letters-test-labels-idx1-ubyte.gz",
                                 transform=transform_train)
    else:
        data_train = LocalDataset("../../../Data/RecognitionData/EMNIST_DATA/",
                                  "emnist-digits-train-images-idx3-ubyte.gz",
                                  "emnist-digits-train-labels-idx1-ubyte.gz",
                                  transform=transform_train)
        data_test = LocalDataset("../../../Data/RecognitionData/EMNIST_DATA/",
                                 "emnist-digits-test-images-idx3-ubyte.gz",
                                 "emnist-digits-test-labels-idx1-ubyte.gz",
                                 transform=transform_train)

    train_loader = DataLoader(
        dataset=data_train,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    if modelType is "letters":
        model = AlexNet(args.channels).to(device)
    else:
        model = FC_Model().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step(epoch)

    if args.save_model:
        if modelType is "letters":
            torch.save(model, "../AlphabetRecognitPerTrainModel/AlexNet_letter_new.pth")
        elif modelType is "digits":
            torch.save(model, "../AlphabetRecognitPerTrainModel/FC_Model_new.pth")
        else:
            pass
