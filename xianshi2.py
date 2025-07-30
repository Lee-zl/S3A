
from scipy import io
import spectral
import numpy as np
import wx
import random
img = np.zeros((610,340,103))
weight = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            img[i,j,k] = random.random()
            weight[i,j,k] = 0.5

#data = io.loadmat(r'/home/gpx/ELR/ELR-master/ELR/Datasets/PaviaU/PaviaU.mat')['paviaU'] # 两个xxx按需要换成自己的
data = img*weight
vmax = data.max(axis=(0, 1))
vmin = data.min(axis=(0, 1))
vv = (data - vmin) / (vmax - vmin)*255
vv = vv.astype(np.uint8)
app = wx.App()
spectral.settings.WX_GL_DEPTH_SIZE=16
#spectral.view_cube(vv, bands=[59,38,20])	# bands参数按需要换成自己的
spectral.view_cube(vv)
app.MainLoop()
