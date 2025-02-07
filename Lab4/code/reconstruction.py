#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel):
    tree = KDTree(points)
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                x = min_grid[0] + i * size_voxel
                y = min_grid[1] + j * size_voxel
                z = min_grid[2] + k * size_voxel
                grid_point = np.array([x, y, z])
                
                dist, index = tree.query(grid_point.reshape(1, -1), k=1)
                closest_point = points[index[0][0]]
                normal = normals[index[0][0]]
                
                scalar_field[i, j, k] = np.dot(normal, grid_point - closest_point)
    

# IMLS surface reconstruction
def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):
    h = 0.01
    tree = KDTree(points)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                x = min_grid[0] + i * size_voxel
                y = min_grid[1] + j * size_voxel
                z = min_grid[2] + k * size_voxel
                grid_point = np.array([x, y, z])

                dists, indices = tree.query(grid_point.reshape(1, -1), k=knn)
                dists = dists[0]
                indices = indices[0]

                vec_diffs = grid_point - points[indices]  # (knn,3)
                
                dot_products = np.sum(normals[indices] * vec_diffs, axis=1)
                
                weights = np.exp(- (dists ** 2) / (h ** 2)).reshape(-1, 1)

                num = np.sum(dot_products * weights.flatten())
                denom = np.sum(weights)

                scalar_field[i, j, k] = num / denom


if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128 #128
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
    print("size_voxel: ", size_voxel)
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

	# Compute the scalar field in the grid

    choice = "imls"

    if choice == "hoppe":
        compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    elif choice == "imls":
        compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel,size_voxel,size_voxel))
    verts += min_grid
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)

    num_triangles = len(faces)

    print("Total number of triangles:", num_triangles)

    mesh.export(file_obj=f'../bunny_mesh_{choice}_{grid_resolution}.ply', file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	


