import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio

# read image 
def simple_image_read(filename):
    with open(filename,"rb") as reader:
        # read header information
        image_size = reader.read(3*4)
        voxdims = reader.read(3*4)
        # read image intensity
        image = reader.read()
        # convert byte into float32 or int16
        image_size = np.frombuffer(image_size, np.float32)
        voxdims = np.frombuffer(voxdims, np.float32)
        image = np.frombuffer(image, np.int16)
        image = image.reshape(224, 224, 30)
    return image, image_size, voxdims

# write image, firt header information, then image intensity
def simple_image_write(image, header, filename):
    # convert int32, float64 into int16, float32
    image = image.astype(np.int16)
    header = header.astype(np.float32)
    writer = open(filename,"wb")
    writer.write(header.tobytes())
    writer.write(image.tobytes())
    writer.close()
        
    
if __name__ == '__main__':
    image = scio.loadmat("example_image.mat")["vol"]
    voxdims = scio.loadmat("example_image.mat")["voxdims"].squeeze()
    # generate header information
    header = np.array([np.size(image,0),np.size(image,1),np.size(image,2),voxdims[0],voxdims[1],voxdims[2]])
    filename = "image.sim"
    # write array to sim file
    simple_image_write(image, header, filename)
    # read sim as array
    image, image_size, voxdims = simple_image_read(filename)
    # show results
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
    
    
    
    