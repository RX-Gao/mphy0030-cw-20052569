import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# find neighbors from triangles and vertice number
def find_neighbors(triangles, vertice):
    positions = np.where(triangles==vertice)
    # find triangles which contain the vertice number
    neighbors = triangles[positions[0],:]
    # calculate positions of the vertice in all neighbor triangles
    index = np.arange(len(positions[1]))*3 + positions[1]
    # delete the vertice and repeated neighbors to generate all neighbors' numbers of the vertice
    neighbors = np.delete(neighbors, index)
    neighbors = np.unique(neighbors)
    return neighbors.astype(int)

#  lowpass mesh smoothing
def lowpass_mesh_smoothing(triangles, vertices, iterations=10, Lambda=0.9, Mu=-1.02*0.9):
    first_vertices = np.copy(vertices)
    second_vertices = np.copy(vertices)
    # iterate mesh smoothing algorithm
    for iteration in range(iterations):
        # mesh smoothing for each verticle
        for i in range(np.size(vertices,0)):
            # find all neighbors' numbers of the ith verticle 
            neighbors = find_neighbors(triangles, i)
            # calculate correspinding neighbors' coordinates of the ith verticle
            neighbors_coordinate = vertices[neighbors,:]
            # do the first mesh smoothing according to paper
            first_vertices[i,:] = vertices[i,:] + Lambda/len(neighbors)* \
                               np.sum(neighbors_coordinate - vertices[i,:], axis=0)
            # update neighbors' coordinates of the ith verticle after the first mesh smoothing
            neighbors_coordinate = first_vertices[neighbors,:]
            # do the second mesh smoothing according to paper
            second_vertices[i,:] = first_vertices[i,:] + Mu/len(neighbors)* \
                               np.sum(neighbors_coordinate - first_vertices[i,:], axis=0)
        # update filtered vertices            
        vertices = second_vertices
    return vertices
        
   
if __name__ == '__main__':
    triangles = np.genfromtxt("example_triangles.csv",delimiter=',')-1
    vertices = np.genfromtxt("example_vertices.csv",delimiter=',')
    
    # smooth the surface mesh with three different numbers of iterations, 5, 10, 25
    vertices_5 = lowpass_mesh_smoothing(triangles, vertices, iterations=5)
    vertices_10 = lowpass_mesh_smoothing(triangles, vertices_5, iterations=10-5)
    vertices_25 = lowpass_mesh_smoothing(triangles, vertices_10, iterations=25-10)
    
    # plot results
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=triangles, color='b', alpha=0.3)
    plt.title("0 iteration", fontsize=15)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(vertices_5[:,0], vertices_5[:,1], vertices_5[:,2], triangles=triangles, color='b', alpha=0.3)
    plt.title("5 iterations", fontsize=15)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(vertices_10[:,0], vertices_10[:,1], vertices_10[:,2], triangles=triangles, color='b', alpha=0.3)
    plt.title("10 iterations", fontsize=15)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(vertices_25[:,0], vertices_25[:,1], vertices_25[:,2], triangles=triangles, color='b', alpha=0.3)
    plt.title("25 iterations", fontsize=15)
    plt.show()
    
    
