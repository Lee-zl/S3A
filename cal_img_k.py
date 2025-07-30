from scipy import io
import wx
import numpy as np
import spectral
import sklearn
import sklearn.model_selection

def cal_img_k():
    #img:读取原高光谱图像
    mat_name = 'PaviaU.mat'
    file_name = '/home/gpx/ELR/ELR-master/ELR/Datasets/PaviaU/'
    img = io.loadmat(file_name + mat_name)['paviaU']# Load image to numpy.ndarray
    img = (img - np.min(img))/(np.max(img)-np.min(img))     # Normalization

    #img2:读取权重cube
    mat_name2 = 'saveddata.mat'
    file_name2 = '/home/gpx/ELR/ELR-master/ELR/'
    img2 = io.loadmat(file_name2 + mat_name2)['cube']# Load image to numpy.ndarray
    img2 = (img2 - np.min(img2))/(np.max(img2)-np.min(img2))     # Normalization

    #权重cube可视化
    '''
    vmax = img.max(axis=(0, 1))
    vmin = img.min(axis=(0, 1))
    vv = (img - vmin) / (vmax - vmin)*255
    vv = vv.astype(np.uint8)
    app = wx.App()
    spectral.settings.WX_GL_DEPTH_SIZE=16
    #spectral.view_cube(vv, bands=[59,38,20])	# bands参数按需要换成自己的
    spectral.view_cube(vv)
    app.MainLoop()
    '''

    #计算高光谱图像每个像素K最优波段组成的图像img_k:np.array(610,340,5)
    best_k = 5
    img_k = np.zeros((img2.shape[0],img2.shape[1],best_k))
    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            sortIndex = img2[x,y,:].argsort()
            img[x,y,:] = img[x,y,:][sortIndex[::-1]]
            img_k[x,y,:] = img[x,y,:][0:best_k]

    return img_k
