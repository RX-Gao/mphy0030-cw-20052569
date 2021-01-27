import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def gaussian_pdf(x, mean, COV):
    xhat = x - mean
    p = np.zeros(np.size(x,1))
    for i in range(np.size(x,1)):
        p[i] = np.exp(-1/2*xhat[:,i].transpose()@np.linalg.inv(COV)@xhat[:,i]) \
            /(pow(2*math.pi,3/2) * math.sqrt(np.linalg.det(COV)))
#    p = np.diag(np.exp(-1/2*xhat.transpose()@np.linalg.inv(COV)@xhat) \
#        /(pow(2*math.pi,3/2) * math.sqrt(np.linalg.det(COV))))
    return p

if __name__ == '__main__':
    X = np.random.rand(3, 10000)
    mean = np.mean(X, axis=1).reshape(3, 1) 
    COV = np.cov(X)  
    
    N = 100
    t = np.linspace(-np.pi, np.pi, N)
    r = np.linspace(0, 1, N)
    x1 = np.outer(np.cos(t), r) + mean[0]
    x1 = np.expand_dims(x1,2).repeat(N,axis=2)
    x2 = np.outer(np.sin(t), r) + mean[1]
    x2 = np.expand_dims(x2,2).repeat(N,axis=2)
    x3 = np.linspace(-1, 1, N) + mean[2]
    x3 = np.expand_dims(x3,0).repeat(N,axis=0)
    x3 = np.expand_dims(x3,1).repeat(N,axis=1)
    X = np.concatenate((x1.reshape(1,-1), x2.reshape(1,-1), x3.reshape(1,-1)),axis=0)
    
    p = gaussian_pdf(X, mean, COV)
    
    p_list = [0.1,0.5,0.9]
    color_dict = {0.1:'#ADD8E6', 0.5:'#00BFFF', 0.9:'#4682B4'}
    title_dict = {0.1:'10th percentile', 0.5:'50th percentile', 0.9:'90th percentile'}
    fig = plt.figure()
    ax = Axes3D(fig)
    for pi in p_list:
        percentile = max(p)*pi       
        position = np.where(np.abs(p-percentile)<0.03)
        x1, x2, x3 = X[0,position], X[1,position], X[2,position]
        ax.scatter(x1, x2, x3, color=color_dict[pi], alpha=0.5, s=15)
        
        a = (x1.max() - x1.min())/2
        b = (x2.max() - x2.min())/2
        c = (x3.max() - x3.min())/2
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = a * np.outer(np.cos(u), np.sin(v)) + mean[0]
        y = b * np.outer(np.sin(u), np.sin(v)) + mean[1]
        z = c * np.outer(np.ones(np.size(u)), np.cos(v)) + mean[2]
        
        #Plot the surface
        fig1 = plt.figure()
        ax1 = Axes3D(fig1)
        ax1.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.8)
        ax1.set_xlabel('x1')
        ax1.set_xlim(-0.1, 1.1)
        ax1.set_ylabel('x2')
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_zlabel('x3')
        ax1.set_zlim(-0.1, 1.1)
        ax1.set_title(title_dict[pi], fontsize=15)
    ax.set_xlabel('x1')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylabel('x2')
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlabel('x3')
    ax.set_zlim(-0.1, 1.1)
    ax.set_title("three ellipsoid surfaces", fontsize=15)
    plt.show()
   
