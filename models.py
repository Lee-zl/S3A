import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class CNN_2D(nn.Module):

    def __init__(self, input_channels = 220, n_classes=16, dropout=True):
        super(CNN_2D, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(64)

        self.fc2 = nn.Linear(128, n_classes)
        self.bn4 = nn.BatchNorm1d(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)


        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpooling(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # x = self.bn4(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class TGMandTRM(nn.Module):
    def __init__(self, h, channel, rank,norm_layer=None):
        super(TGMandTRM, self).__init__()
        self.rank = rank
        self.ps = [1, 1, 1, 1]
        self.h = h
        conv1_1, conv1_2, conv1_3 = self.ConvGeneration(self.rank, h, channel)
        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3

        self.lam = torch.ones(self.rank, requires_grad=True).cuda()

        self.pool = nn.AdaptiveAvgPool2d(self.ps[0])

        # self.fusion = nn.Sequential(
        #     nn.Conv2d(512, 512, 1, padding=0, bias=False),
        #     norm_layer(512),
        #     # nn.Sigmoid(),
        #     nn.ReLU(True),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')


    def forward(self, x):
        b, c, height, width = x.size()
        C = self.pool(x)
        H = self.pool(x.permute(0, 3, 1, 2).contiguous())
        W = self.pool(x.permute(0, 2, 3, 1).contiguous())
        self.lam = F.softmax(self.lam,-1)
        lam = torch.chunk(self.lam, dim=0, chunks=self.rank)
        list = []
        for i in range(0, self.rank):
            list.append(lam[i]*self.TukerReconstruction(b, self.h , self.ps[0], self.conv1_1[i](C), self.conv1_2[i](H), self.conv1_3[i](W)))
        tensor1 = sum(list)
        # tensor1 = torch.cat((x , F.relu_(x * tensor1)), 1)
        tensor1 = x + F.relu_(x * tensor1)
        return tensor1

    def ConvGeneration(self, rank, h,channel):
        conv1 = []
        n = 1
        for _ in range(0, rank):
                conv1.append(nn.Sequential(
                nn.Conv2d(channel, channel // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, rank):
                conv2.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, rank):
                conv3.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv3 = nn.ModuleList(conv3)

        return conv1, conv2, conv3

    def TukerReconstruction(self, batch_size, h, ps, feat, feat2, feat3):
        b = batch_size
        C = feat.view(b, -1, ps)
        H = feat2.view(b, ps, -1)
        W = feat3.view(b, ps * ps, -1)
        CHW = torch.bmm(torch.bmm(C, H).view(b, -1, ps * ps), W).view(b, -1, h, h)
        return CHW


class Tensor_Reconet(nn.Module):

    def __init__(self, input_channels = 220, n_classes=16, l = True, m = True, h = True,dropout=True):
        super(Tensor_Reconet, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.atten1 = TGMandTRM(27,16, 8)
        self.l = l
        self.conv2 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.atten2 = TGMandTRM(13,32, 8)
        self.m = m
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpooling = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.atten3 = TGMandTRM(6 ,64, 8)
        self.h = h
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc2 = nn.Linear(64, n_classes)
        self.bn4 = nn.BatchNorm1d(64)
    def forward(self, x):
            x_input1_conv = self.conv1(x)
            x_input1_conv = self.relu(x_input1_conv)
            if self.l == True:
                x_input1_att = self.atten1(x_input1_conv)
            else:
                x_input1_att = 0
            x_input2 = self.conv2(x)
            x_input2 = self.relu(x_input2)
            x = x_input1_conv + x_input2 + x_input1_att
            x = self.bn1(x)
            x = self.maxpooling(x)


            x_input1_conv = self.conv3(x)
            x_input1_conv = self.relu(x_input1_conv)
            if self.m == True:
                x_input1_att = self.atten2(x_input1_conv)
            else:
                x_input1_att = 0
            x_input2 = self.conv4(x)
            x_input2 = self.relu(x_input2)
            x = x_input1_conv + x_input2 + x_input1_att
            x = self.bn2(x)
            x = self.maxpooling(x)


            x_input1_conv = self.conv5(x)
            x_input1_conv = self.relu(x_input1_conv)
            if self.h == True:
                x_input1_att = self.atten3(x_input1_conv)
            else:
                x_input1_att = 0
            x_input2 = self.conv6(x)
            x_input2 = self.relu(x_input2)
            x = x_input1_conv + x_input2 + x_input1_att
            x = self.avgpool(x)

            output_feature = x
            x = x.view(x.size(0), -1)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x, output_feature

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x),y

class SE_Net(nn.Module):

    def __init__(self, input_channels = 220, n_classes=16, dropout=True):
        super(SE_Net, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.SE = SELayer(input_channels)

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc2 = nn.Linear(64, n_classes)
        self.bn4 = nn.BatchNorm1d(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x, channel_weight = self.SE(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpooling(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        out_feature = x

        x = x.view(x.size(0), -1)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x,channel_weight,out_feature