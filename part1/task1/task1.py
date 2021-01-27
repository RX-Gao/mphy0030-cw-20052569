import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio


def simple_image_read(filename):
    with open(filename,"rb") as reader:
        image_size = reader.read(3*4)
        voxdims = reader.read(3*4)
        image = reader.read()
        image_size = np.frombuffer(image_size, np.float32)
        voxdims = np.frombuffer(voxdims, np.float32)
        image = np.frombuffer(image, np.int16)
        image = image.reshape(224, 224, 30)
    return image, image_size, voxdims

def simple_image_write(image, header, filename):
    image = image.astype(np.int16)
    header = header.astype(np.float32)
    writer = open(filename,"wb")
    writer.write(header.tobytes())
    writer.write(image.tobytes())
    writer.close()
        
    
if __name__ == '__main__':
    image = scio.loadmat("example_image.mat")["vol"]
    voxdims = scio.loadmat("example_image.mat")["voxdims"].squeeze()
    header = np.array([np.size(image,0),np.size(image,1),np.size(image,2),voxdims[0],voxdims[1],voxdims[2]])
    filename = "image.sim"
    simple_image_write(image, header, filename)
    image, image_size, voxdims = simple_image_read(filename)
    print("image size: %d %d %d\nvoxdims: %f %f %f" %(image_size[0],image_size[1], image_size[2], \
                                                    voxdims[0], voxdims[1], voxdims[2]))
    plt.figure(figsize = (20,20))
    plt.subplot(311)
    plt.imshow(image[:,:,0], cmap ='gray')
    plt.subplot(312)
    plt.imshow(image[:,:,10], cmap ='gray')
    plt.subplot(313)
    plt.imshow(image[:,:,20], cmap ='gray')
    plt.show()
    
    
    
    