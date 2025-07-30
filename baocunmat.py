from scipy import io 
import matplotlib.pyplot as plt
import numpy as np
#python创建一个mat文件
x = np.zeros((5,5,5))
io.savemat('/home/gpx/ELR/ELR-master/ELR/saveddata.mat', {'x':x})  #变量分别保存在名字为xyz下面
