import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def find_neighbors(triangles, vertice):
    positions = np.where(triangles==vertice)
    neighbors = triangles[positions[0],:]
    index = np.arange(len(positions[1]))*3 + positions[1]
    neighbors = np.delete(neighbors, index)
    neighbors = np.unique(neighbors)
    return neighbors.astype(int)
  
def lowpass_mesh_smoothing(triangles, vertices, iterations=10, Lambda=0.9, Mu=-1.02*0.9):
    first_vertices = np.copy(vertices)
    second_vertices = np.copy(vertices)
    for iteration in range(iterations):
        for i in range(np.size(vertices,0)):
            neighbors = find_neighbors(triangles, i)
            neighbors_coordinate = vertices[neighbors,:]
            first_vertices[i,:] = vertices[i,:] + Lambda/len(neighbors)* \
                               np.sum(neighbors_coordinate - vertices[i,:], axis=0)
            neighbors_coordinate = first_vertices[neighbors,:]
            second_vertices[i,:] = first_vertices[i,:] + Mu/len(neighbors)* \
                               np.sum(neighbors_coordinate - first_vertices[i,:], axis=0)
        vertices = second_vertices
    return vertices
        
   
if __name__ == '__main__':
    triangles = np.genfromtxt("example_triangles.csv",delimiter=',')-1
    vertices = np.genfromtxt("example_vertices.csv",delimiter=',')
    
    vertices_5 = lowpass_mesh_smoothing(triangles, vertices, iterations=5)
    vertices_10 = lowpass_mesh_smoothing(triangles, vertices_5, iterations=10-5)
    vertices_25 = lowpass_mesh_smoothing(triangles, vertices_10, iterations=25-10)
    
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
    
    
