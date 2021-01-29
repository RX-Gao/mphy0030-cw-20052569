import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
import scipy.io as scio


class RBFSpline:  
    def __init__(self, sigma, LAMBDA):
        self.sigma = sigma
        self.LAMBDA = LAMBDA
       
    # calculate K between query and control points, vectorisation strategies are proved in report
    def kernel_gaussian(self, query, control):
        pi = np.expand_dims(query,1).repeat(control.shape[0],axis=1)
        pj = np.expand_dims(control,0).repeat(query.shape[0],axis=0)
        r = np.linalg.norm((pi-pj).astype(np.float32),axis=2)
        R = lambda r: np.exp(-1/2*(r/self.sigma)**2)
        K = R(r)
        return K
    
    # fit alpha between fixed control and moved control points
    def fit(self, source, target):
        # calculate K between source points, namely control points
        K = self.kernel_gaussian(source, source)
        # approximate registration
        K += np.diag(np.ones(K.shape[0])*self.LAMBDA)
        # register between source and target pionts, namely fixed control and moved points respectively
        # solve registration parameters by using SVD, detailed proof are elaborated in report
        u, s, vh = np.linalg.svd(K)
        s = np.diag(s)
        alpha = vh.T@np.linalg.inv(s)@u.T@target
        return alpha
        
    # deform query points according to deformation of control points
    def evaluate(self, query, control, alpha):
        # calculate K between query and control points
        K = self.kernel_gaussian(query, control)
        # calculate transformed query points 
        transformedQuery = np.dot(K, alpha)
        return transformedQuery 


class Image3D:
    # initialize and assign for all useful information of Image
    def __init__(self, Image):
        self.image = Image["vol"]
        self.voxdims = Image["voxdims"].squeeze()
        self.size = self.image.shape
        x, y, z = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), np.arange(self.size[2]), indexing='ij')
        self.points = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)),axis=1) * self.voxdims


class FreeFormDeformation:    
    # initilize all kinds of points used in this class but not assign for them 
    def __init__(self):
        self.control = None
        self.control_moving = None
        self.query = None
    
    # unused constructor, which has similar function with the optional one
    def constructor(self, NUM, RANGE):
        x_control = np.ceil(np.linspace(RANGE[0,0], RANGE[0,1], NUM[0], endpoint = False))
        y_control = np.ceil(np.linspace(RANGE[1,0], RANGE[1,1], NUM[1], endpoint = False))
        z_control = np.ceil(np.linspace(RANGE[2,0], RANGE[2,1], NUM[2], endpoint = False))
        x, y, z = np.meshgrid(x_control, y_control, z_control, indexing='ij')
        self.control = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)) ,axis=1)
        self.control_moving = self.control
        x, y, z = np.meshgrid(np.arange(RANGE[0,0], RANGE[0,1]), np.arange(RANGE[1,0], RANGE[1,1]),                               np.arange(RANGE[2,0], RANGE[2,1]), indexing='ij')
        self.query = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)),axis=1)

    # constructor, generate fixed and moved control points and query pints
    # detailed control points selection method is described in report
    # assign for initialized class variables 
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
     
    # interpolate transformed query points to generate warped image
    def __Point2Image(self, transformedQuery, Image3D):
        points = (np.arange(Image3D.size[0])*Image3D.voxdims[0], np.arange(Image3D.size[1])*Image3D.voxdims[1], np.arange(Image3D.size[2])*Image3D.voxdims[2])
        values = Image3D.image
        transformedImage = interpn(points, values, transformedQuery, bounds_error = False, fill_value = 0).reshape(Image3D.size)
        return transformedImage
    
    # biophysically plausible random deformation, detailed description are in report 
    def _truncated_normal(self, mean, stddev, truncate, size):
        return np.clip(np.random.normal(mean, stddev, size), -truncate, truncate) 
  
    def __random_transform_generator(self,  randomness):
        transform = np.diag(np.ones(3)) + self._truncated_normal(0, randomness/5, randomness/5, (3, 3))
        self.control_moving = np.dot(self.control, transform)
    
    # warp image using RBFSpline class, detailed description are in report
    def __warp_image(self, Image3D, RBFSpline):
        # fit RBF Spine parameters between fixed control points and moved control points
        alpha = RBFSpline.fit(self.control, self.control_moving)
        # evaluate RBF Spine to get transformed query points
        transformedQuery = RBFSpline.evaluate(self.query, self.control, alpha)
        # convert points to image
        warpImage = self.__Point2Image(transformedQuery, Image3D)
        return warpImage
    
    # transform image randomly
    def random_transform(self, Image3D, RBFSpline, randomness):
        # generate randomly transformed control points(moved control points) according to randomness
        self.__random_transform_generator(randomness)
        # interpolate query points from control points to get warp image
        warpImage = self.__warp_image(Image3D, RBFSpline)
        return warpImage


if __name__ == '__main__':
      # image load 
      Image = scio.loadmat("example_image.mat")
      Image3D_object = Image3D(Image)
      
      # parameters definition
      sigma, LAMBDA = 40, 0.001
      NUM = [3, 3, 3]
      randomness = 0.5
      
      # warp image
      RBFSpline = RBFSpline(sigma, LAMBDA)
      FreeFormDeformation_object = FreeFormDeformation()
      FreeFormDeformation_object.constructor_opt(Image3D_object, NUM)
      warpImage = FreeFormDeformation_object.random_transform(Image3D_object, RBFSpline, randomness)
      
      # plot original and warped slices
      plt.figure(figsize = (15,15))
      plt.subplot(211)
      plt.imshow(Image3D_object.image[:,:,15], cmap ='gray')
      plt.title("original image slice")
      plt.subplot(212)
      plt.imshow(warpImage[:,:,15], cmap ='gray')
      plt.title("warped image slice")
      plt.savefig('sclices.png')
      plt.show()






