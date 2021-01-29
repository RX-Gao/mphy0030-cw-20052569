import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
import scipy.io as scio


class RBFSpline:  
    def __init__(self, sigma, LAMBDA):
        self.sigma = sigma
        self.LAMBDA = LAMBDA
        
    def kernel_gaussian(self, query, control):
        pi = np.expand_dims(query,1).repeat(control.shape[0],axis=1)
        pj = np.expand_dims(control,0).repeat(query.shape[0],axis=0)
        r = np.linalg.norm((pi-pj).astype(np.float32),axis=2)
        R = lambda r: np.exp(-1/2*(r/self.sigma)**2)
        K = R(r)
        return K
    
    def fit(self, source, target):
        K = self.kernel_gaussian(source, source)
        K += np.diag(np.ones(K.shape[0])*self.LAMBDA)
        u, s, vh = np.linalg.svd(K)
        s = np.diag(s)
        alpha = vh.T@np.linalg.inv(s)@u.T@target
        return alpha
        
    def evaluate(self, query, control, alpha):
        K = self.kernel_gaussian(query, control)
        transformedQuery = np.dot(K, alpha)
        return transformedQuery 


class Image3D:
    def __init__(self, Image):
        self.image = Image["vol"]
        self.voxdims = Image["voxdims"].squeeze()
        self.size = self.image.shape
        x, y, z = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), np.arange(self.size[2]), indexing='ij')
        self.points = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)),axis=1) * self.voxdims


class FreeFormDeformation:    
    def __init__(self):
        self.control = None
        self.control_moving = None
        self.query = None
    
    def constructor(self, NUM, RANGE):
        x_control = np.ceil(np.linspace(RANGE[0,0], RANGE[0,1], NUM[0], endpoint = False))
        y_control = np.ceil(np.linspace(RANGE[1,0], RANGE[1,1], NUM[1], endpoint = False))
        z_control = np.ceil(np.linspace(RANGE[2,0], RANGE[2,1], NUM[2], endpoint = False))
        x, y, z = np.meshgrid(x_control, y_control, z_control, indexing='ij')
        self.control = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)) ,axis=1)
        self.control_moving = self.control
        x, y, z = np.meshgrid(np.arange(RANGE[0,0], RANGE[0,1]), np.arange(RANGE[1,0], RANGE[1,1]),                               np.arange(RANGE[2,0], RANGE[2,1]), indexing='ij')
        self.query = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)),axis=1)

    def constructor_opt(self, Image3D, NUM):
        x_control, y_control, z_control = np.zeros(NUM[0]), np.zeros(NUM[1]), np.zeros(NUM[2])
        x_control[0], x_control[NUM[0]-1] = 0, Image3D.size[0]-1
        y_control[0], y_control[NUM[1]-1] = 0, Image3D.size[1]-1
        z_control[0], z_control[NUM[2]-1] = 0, Image3D.size[2]-1
        x_control[1:NUM[0]-1] = np.ceil(np.linspace(1, Image3D.size[0]-1, NUM[0]-1, endpoint = False))[1:]
        y_control[1:NUM[1]-1] = np.ceil(np.linspace(1, Image3D.size[1]-1, NUM[1]-1, endpoint = False))[1:]
        z_control[1:NUM[2]-1] = np.ceil(np.linspace(1, Image3D.size[2]-1, NUM[2]-1, endpoint = False))[1:]
        x, y, z  = np.meshgrid(x_control, y_control, z_control, indexing='ij')
        self.control = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)),axis=1) * Image3D.voxdims
        self.control_moving = self.control
        self.query = Image3D.points
               
    def __Point2Image(self, transformedQuery, Image3D):
        points = (np.arange(Image3D.size[0])*Image3D.voxdims[0], np.arange(Image3D.size[1])*Image3D.voxdims[1],                   np.arange(Image3D.size[2])*Image3D.voxdims[2])
        values = Image3D.image
        transformedImage = interpn(points, values, transformedQuery, bounds_error = False, fill_value = 0).reshape(Image3D.size)
        return transformedImage

    def _truncated_normal(self, mean, stddev, truncate, size):
        return np.clip(np.random.normal(mean, stddev, size), -truncate, truncate) 
  
    def __random_transform_generator(self,  randomness):
        transform = np.diagflat(np.array([1,1,1])) + self._truncated_normal(0, randomness/5, randomness/5, (3, 3))
        print(transform)
        self.control_moving = np.dot(self.control, transform)
  
    def __warp_image(self, Image3D, RBFSpline):
        alpha = RBFSpline.fit(self.control, self.control_moving)
        transformedQuery = RBFSpline.evaluate(self.query, self.control, alpha)
        warpImage = self.__Point2Image(transformedQuery, Image3D)
        return warpImage
      
    def random_transform(self, Image3D, RBFSpline, randomness):
        self.__random_transform_generator(randomness)
        warpImage = self.__warp_image(Image3D, RBFSpline)
        return warpImage


if __name__ == '__main__':
      #image load 
      Image = scio.loadmat("example_image.mat")
      Image3D_object = Image3D(Image)
      #parameters definition
      sigma, LAMBDA = 30, 0.001
      NUM = [4, 4, 4]
      randomness = 0.5
      #warp image
      RBFSpline = RBFSpline(sigma, LAMBDA)
      FreeFormDeformation_object = FreeFormDeformation()
      FreeFormDeformation_object.constructor_opt(Image3D_object, NUM)
      warpImage = FreeFormDeformation_object.random_transform(Image3D_object, RBFSpline, randomness)
      #plot
      plt.figure()
      plt.imshow(Image3D_object.image[:,:,15], cmap ='gray')
      plt.figure()
      plt.imshow(warpImage[:,:,15], cmap ='gray')
      plt.show()






