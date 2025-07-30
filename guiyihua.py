import numpy as np
def Norm(vector, min, max):
    for i in range(len(vector)):
        vector[i] = (vector[i]-min)/(max - min)
    return vector
dis_space = np.array((100,236,87,20,39))
dis_space = dis_space.astype(np.float64)
dis_space = Norm(dis_space,min(dis_space),max(dis_space))
print('dis_space=',dis_space)