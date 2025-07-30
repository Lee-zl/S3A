import argparse
import collections
from dis import dis
from matplotlib.colors import Normalize
from scipy import io
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.utils import shuffle
from sqlalchemy import true
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import mlflow
import mlflow.pytorch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from collections import OrderedDict
import sklearn
import sklearn.model_selection
import os
import pandas as pd
import spectral
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import random
from torch.backends import cudnn
from models import *
from spectral import *

from scipy.io import loadmat
#建立数据
'''
img = np.zeros((610,340,103))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            img[i,j,k] = 0.5
'''
img = loadmat(r'//home/gpx/ELR/ELR-master/ELR/Datasets/IndianPines/IndianPines_GT.mat')['paviaU']
img2 = loadmat(r'//home/gpx/ELR/ELR-master/ELR/Datasets/PaviaU/PaviaU_gt.mat')
spectral.settings.WX_GL_DEPTH_SIZE = 16
#spectral.view_cube(img)
view_cube(img,bands=[29,19,9])
#obj = mlab.contour3d(img,contours='',transparent=True)
#contours八个等值面　　transparent该对象可以透明表示，可以查看内部
#mlab.show()