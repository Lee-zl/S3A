import argparse
import collections
from dis import dis
from typing import final
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
import wx
import time
device = torch.device('cuda:{}'.format(0))
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)
cudnn.deterministic = True
data_index = []


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))       # 打印按指定格式排版的时间
# 计算光谱距
def calcDis(dataSet, centrobands, k, img):
    #centrobands:num_cluster*103
    clalist=[]
    for data in dataSet: 
        diff = np.tile(img[data[0],data[1],:], (k, 1)) - centrobands  #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2     #平方
        squaredDist = np.sum(squaredDiff, axis=1)   #和(axis=1表示行)   squaredDist:20*1
        distance = squaredDist ** 0.5  #开根号
        clalist.append(distance)
    print('HAHAHAHAHA') 
    clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist

#计算光谱角制图
def cos(array1, array2):
    norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
    norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array2))))
    return sum([array1[i]*array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)

def calcArgDis(dataSet, centrobands, k, img):
    #centrobands:num_cluster*103
    clalist=[]
    for data in dataSet: 
        #diff = 
        #diff = np.tile(img[data[0],data[1],:], (k, 1)) - centrobands  #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDist = []
        for i in range(len(centrobands)):
            # x = math.cos(img[data[0],data[1],:]/centrobands[i])
            # np.tile(img[data[0],data[1],:], (k, 1))
            x = cos(img[data[0],data[1],:],centrobands[i])
            squaredDist.append(np.arccos(x))
        #squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
        distance = squaredDist
        clalist.append(distance)
    print('HiHiHiHiHi') 
    clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist

# 计算质心
def classify(dataSet, centrobands, k, img):
    #dataSet:4w * 2
    clalist = calcDis(dataSet, centrobands, k, img)#clalist:4w * 20,每个点到20个质心的光谱距
    # 分组并计算新的质心,质心为光谱
    minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
    #minDistIndices:4w*1,值为0-19的整数
    
    bandset = []
    for i in range(len(dataSet)):
        xy = dataSet[i]
        bandset.append(img[xy[0],xy[1],:])
    #bandset:4w*103
    newCentrobands = pd.DataFrame(bandset).groupby(minDistIndices).mean() 
    #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentrobands = newCentrobands.values #newCentrobands:num_cluster*103
 
    # 计算变化量
    changed = newCentrobands - centrobands
    print('1111111111')
    return changed, newCentrobands

# 使用k-means分类
def kmeans(dataSet, k, img):
    # 随机取质心,centroids只用一次
    centroids = random.sample(dataSet, k)
    centrobands = []
    centroids = np.array(centroids)
    for i in range(k):
        centrobands.append(img[centroids[i,0],centroids[i,1]])
    #质心为波段而不是空间坐标,centrobands:20*103
    
    # 更新质心 直到变化量全为0
    changed, newCentrobands = classify(dataSet, centrobands, k, img)
    Iter = 0
    #while (np.any(changed != 0) and Iter < 5):
    while np.any(changed != 0):
        changed, newCentrobands = classify(dataSet, newCentrobands, k, img)
        #print('changed=',changed)
        Iter = Iter + 1
    centrobands = sorted(newCentrobands.tolist())   #tolist()将矩阵转换成列表 sorted()排序
 
    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centrobands, k, img) #调用计算光谱距
    minDistIndices = np.argmin(clalist, axis=1)  
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):   #enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
        
    return cluster


class getPUdata():
    def __init__(self):
        super(getPUdata,self).__init__()
        datasets = {
                'PaviaC': {
                    'img': 'Pavia.mat',
                    'gt': 'Pavia_gt.mat'
                    },
                'PaviaU': {
                    'img': 'PaviaU.mat',
                    'gt': 'PaviaU_gt.mat',
                    'sr':'PU_segment_results_100'
                    },
                'KSC': {
                    'img': 'KSC.mat',
                    'gt': 'KSC_gt.mat'
                    },
                'IndianPines': {
                    'img': 'IndianPines.mat',
                    'gt': 'IndianPines_GT.mat',
                    'sr':'segment_results'
                    }}
        dataset_name = 'PaviaU'
        dataset = datasets[dataset_name]
        folder = 'Datasets/' + dataset_name + '/'
        print(folder)
        img = io.loadmat(folder+dataset['img'])['paviaU']# Load image to numpy.ndarray
        # ['indian_pines_corrected']
        img = (img - np.min(img))/(np.max(img)-np.min(img))     # Normalization
        print(type(img),img.shape)
        #view0 = spectral.imshow(data=img,title="img")
        gt = io.loadmat(folder + dataset['gt'])['paviaU_gt']    
        sr = io.loadmat(folder + dataset['sr'])['segmentation_results']    #不知道有啥用
        '''
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                            'Painted metal sheets', 'Bare Soil', 'Bitumen',
                            'Self-Blocking Bricks', 'Shadows']

        N_CLASSES = len(label_values)

        '''
        #以gt中所有非0的点作为中心来裁减图片
        indices = np.nonzero(gt)   #nonzero:图转非0点集合  gt中所有不是0的点坐标构成2*N的数组         
        Record = np.zeros((img.shape[0],img.shape[1])).astype('int')
        X = list(zip(*indices))
        #view1 = spectral.imshow(classes=gt,title="gt")
        #plt.imshow(gt)
        #plt.show()
        #记录每个坐标对应在4w个点中的下标，在统计聚类结果时有用
        for i in range(len(X)):
            Xi = X[i]
            Record[Xi[0]][Xi[1]] = i

        y = gt[indices]         #gt中所有不是0的点label构成1*N的数组y
        train_gt = np.zeros_like(gt)    #全0的HW一样的二维矩阵
        test_gt = np.zeros_like(gt)
        clu_map = np.zeros_like(gt)
        #print('train_gt: ',type(train_gt),train_gt.shape)
        
        #kmeans将点聚成2*num_class类，该步骤与波段选择无关,类别数可修改
        num_cluster = (config['num_classes']-1)*2
        self.num_cluster = num_cluster
        clulabel = np.zeros(len(X)).astype(int)
        cluster = kmeans(X, num_cluster, img) #cluster:num_cluster个聚类label分别对应的空间坐标集[]
        for i in range(num_cluster):    #枚举聚类label
            cluster_i = cluster[i]
            for j in range(len(cluster_i)):
                Point = cluster_i[j] #Point:(x,y)
                clu_map[Point[0],Point[1]] = i+1 #因为除了4w个点，背景也是0，所以需要+1
        #view2 = spectral.imshow(classes=clu_map,title="clu_map")
        #plt.imshow(clu_map)
        #plt.show()
        self.clulabel = clulabel
        #按train_size所占比例随机划分训练测试集
        #train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=0.01, random_state = 0,stratify = y)
        
        #每个聚类簇按0.01比例划分训练集再组成总的训练集
        train_indices = []
        test_indices = []
        clu_train = []
        clu_test = []
        #train_indices把所有训练集点放到一起，clu_train把训练集点每簇放到一个list里
        for i in range(num_cluster):
            cluster_i_tolist = [list(xi) for xi in cluster[i]]
            transpose_data = np.transpose(cluster_i_tolist).tolist()
            yi = gt[transpose_data].tolist()
            cnt = np.zeros(10)
            for itery in range(len(yi)):
                cnt[yi[itery]] = cnt[yi[itery]] + 1
            print('countlabel:',cnt)
            for itercnt in range(len(cnt)):
                if cnt[itercnt] == 1:
                    false_xiabiao = yi.index(itercnt)
                    cluster[i].pop(false_xiabiao)
                    yi.pop(false_xiabiao)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))       # 打印按指定格式排版的时间
            train_indices_i, test_indices_i = sklearn.model_selection.train_test_split(cluster[i], train_size=0.01, random_state = 0,stratify = yi)
            clu_train.append(train_indices_i)
            clu_test.append(test_indices_i)
            for number in train_indices_i:
                train_indices.append(number)
            for number in test_indices_i:
                test_indices.append(number)
        self.cluster = cluster
        self.clu_train = clu_train
        self.clu_test = clu_test
        self.train_indices = train_indices
        self.test_indices = test_indices

        train_indices = list(zip(*train_indices))
        #print('train_indices: ',type(train_indices),len(train_indices))
        test_indices = list(zip(*test_indices))
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]
        self.img = img
        self.gt = gt
        self.train_gt = train_gt
        self.test_gt = test_gt
        self.clu_map =clu_map
        print('haha')
        #img:610*340*103 gt:610*340 train_gt:610*340 test_gt:610*340


class HSIdata(torch.utils.data.Dataset):
    def __init__(self, yuchuli ,gt ,batchsize):
        super(HSIdata, self).__init__()     #super初始化父类
        data=yuchuli.img
        self.data = data
        self.label = gt
        #self.label = gt - 1 #矩阵每个元素值都-1，0变255
        self.label = gt
        #print('self.label=',self.label)

        #裁成27*27的图片
        self.patch_size = 27        
        
        self.data_all_offset = np.zeros((data.shape[0] + self.patch_size - 1, self.data.shape[1] + self.patch_size - 1, self.data.shape[2]))
        self.seg_all_offset = np.zeros((data.shape[0] + self.patch_size - 1, self.data.shape[1] + self.patch_size - 1), dtype = np.int32)
        self.start = int((self.patch_size - 1) / 2)
        self.data_all_offset[self.start:data.shape[0] + self.start, self.start:data.shape[1] + self.start,:] = self.data[:, :, :]
        x_pos, y_pos = np.nonzero(gt)

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        #label:610*340显示每个像素的类别,labels:427*1显示每个非0坐标对的类别
        self.batch_size = batchsize
        self.dataset = np.zeros((len(self.indices),data.shape[2],self.patch_size,self.patch_size))
        #裁减HW为27*27
        for i in range(len(self.indices)):
            (x,y) = self.indices[i]
            data = self.data_all_offset[x:x + self.patch_size, y:y + self.patch_size]
            label = self.label[x, y]    #label是该非0点的类别，只有一个数
            data = np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            label = np.asarray(label, dtype='int64')
            #data = torch.from_numpy(data)
            #label = torch.from_numpy(label)
            self.dataset[i] = data
        self.dataset = torch.from_numpy(self.dataset)
        #self.labels = torch.from_numpy(self.labels)
        #print('HSIdata的init结束了')

    #         np.random.shuffle(self.indices)

    def __len__(self):              #在data_loader时执行__len__和__getitem__
        return len(self.indices)

    def __getitem__(self, index):

        data = self.dataset[index]
        label = self.labels[index]
        indices = self.indices[index]
        return data, label, index, indices



def log_params(conf: OrderedDict, parent_key: str = None):
    for key, value in conf.items():
        if parent_key is not None:
            combined_key = f'{parent_key}-{key}'
        else:
            combined_key = key

        if not isinstance(value, OrderedDict):
            mlflow.log_param(combined_key, value)
        else:
            log_params(value, combined_key)

def Norm(vector, min, max):
    for i in range(len(vector)):
        vector[i] = (vector[i]-min)*1.0/(max - min)
    return vector

def Cal_dis(vector1, vector2, k):
    dist = np.tile(vector1, (k, 1)) - vector2
    dist = dist ** 2     #平方
    dist = np.sum(dist, axis=1)   #和(axis=1表示行)
    return dist

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_cifar10.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--lamb', '--lamb'], type=float, target=('train_loss', 'args', 'lambda')),
        CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
    ]
    config = ConfigParser.get_instance(args, options)
    logger = config.get_logger('train')

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    #main(config)

    yuchuli = getPUdata()
    #yuchuli.img:610*340*103 gt:610*340 train_gt:610*340(取gt中不为0的点的1%) test_gt:610*340

    dataset = HSIdata(yuchuli,yuchuli.train_gt,batchsize=64)
    #data_loader = getattr(module_data, config['data_loader']['type'])(
    #    dataset,batch_size=32,shuffle=True,num_batches=0,training=True,num_workers=8,pin_memory=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,batch_size=64,shuffle=True,num_workers=8,pin_memory=True)

    test_dataset = HSIdata(yuchuli,yuchuli.gt,batchsize=128)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,batch_size=128,shuffle=False,num_workers=2)
    valid_data_loader = test_data_loader

    # test_data_loader = None


    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    #import model.model as module_arch

    # get function handles of loss and metrics
    logger.info(config.config)

    num_examp = len(dataset.indices)
    train_loss = getattr(module_loss, config['train_loss']['type'])(num_examp=num_examp, num_classes=config['num_classes'],
                                                                beta=config['train_loss']['args']['beta'])

    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)#调整学习率

    #print('model=',model)
    trainer = Trainer(model, train_loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      val_criterion=val_loss)
    
    trainer.train()
    cube = trainer.cube
    #cube:610*340*103
    #绘制光谱注意力cube
    for clus in range(yuchuli.num_cluster):
        list_train = yuchuli.clu_train[clus]
        list_test = yuchuli.clu_test[clus]
        for i in range(len(list_test)):
            dis_space = []
            dis_band = []
            train_bands = []
            train_spaces = list_train
            x0 = list_test[i][0]
            y0 = list_test[i][1]
            for j in range(len(list_train)):
                train_bands.append(yuchuli.img[list_train[j][0],list_train[j][1],:])
            
            dis_space = Cal_dis(list_test[i], train_spaces, len(list_train))
            dis_space = dis_space.astype(np.float64)
            dis_band = Cal_dis(yuchuli.img[x0,y0,:], train_bands, len(list_train))
            
            #dis_space.append(Cal_dis(list_test[i],list_train[j]))
            #dis_band.append(Cal_dis(list_test[i],list_train[j]))
            
            dis_space = Norm(dis_space,min(dis_space),max(dis_space))
            dis_band = Norm(dis_band,min(dis_band),max(dis_band))
            #dis_space:len(list_train)*1    dis_band:len(list_train)*1
            for j in range(len(list_train)):
                lamb1 = 1
                lamb2 = 1
                weight = 1/(np.sqrt(lamb1 * (dis_space[j]**2) + lamb2 * (dis_band[j]**2))+0.1)
                if (np.sqrt(lamb1 * (dis_space[j]**2) + lamb2 * (dis_band[j]**2))) == 0 :
                    print('weight = INF!!!',x0,y0,list_train[j][0],list_train[j][1],dis_space[j],dis_band[j])
                #print('weight=',weight,' cube=',cube[list_train[j][0],list_train[j][1],:])
                cube[x0,y0,:] = cube[x0,y0,:] + weight * cube[list_train[j][0],list_train[j][1],:]
            #cube:610 * 340 * 103, 即要求的注意力立方
    
    io.savemat('/home/gpx/ELR/ELR-master/ELR/saveddata.mat', {'cube':cube})  #变量分别保存在名字为xyz下面    

    final_img = yuchuli.img*cube
    
    #final_img = yuchuli.img
    vmax = final_img.max(axis=(0, 1))
    vmin = final_img.min(axis=(0, 1))
    vv = (final_img - vmin) / (vmax - vmin)*255
    vv = vv.astype(np.uint8)
    app = wx.App()
    spectral.settings.WX_GL_DEPTH_SIZE=16
    #spectral.view_cube(vv, bands=[59,38,20])	# bands参数按需要换成自己的
    spectral.view_cube(vv)
    app.MainLoop()
    spectral.view_cube(final_img)
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']
