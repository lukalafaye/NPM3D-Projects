#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from numpy import linalg as LA


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points):
    """
    Compute PCA on a set of points.
    
    Parameters:
    - points: (N, 3) numpy array representing the points.
    
    Returns:
    - eigenvalues: (3,) numpy array representing the eigenvalues.
    - eigenvectors: (3, 3) numpy array whose columns are the eigenvectors.
    """

    eigenvalues = None
    eigenvectors = None

    N = points.shape[0]

    centroid = np.sum(points, axis=0).reshape(1, 3) / N

    centered_points = points - centroid

    C = (1 / N) * (centered_points.T @ centered_points)
    
    eigenvalues, eigenvectors = LA.eigh(C)

    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius):
    """
    Compute PCA on neighborhoods of query_points in cloud_points.
    
    Parameters:
    - query_points: (N, 3) numpy array representing the query points.
    - cloud_points: (M, 3) numpy array representing the cloud points.
    - radius: Radius of the neighborhoods.
    
    Returns:
    - all_eigenvalues: (N, 3) numpy array where each row contains the eigenvalues of the PCA.
    - all_eigenvectors: (N, 3, 3) numpy array where each matrix is composed of the eigenvectors of the PCA.
    """

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((query_points.shape[0], 3), dtype=np.float32)
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3), dtype=np.float32)

    tree = KDTree(cloud_points)

    # all_indices = tree.query_radius(query_points, radius)
    all_indices = tree.query(query_points, k=30, return_distance=False)


    for i, indices in enumerate(all_indices):
        if len(indices) > 0:
            #neighbors = cloud_points[indices, :]
            #eigenvalues, eigenvectors = PCA(neighbors)

            #all_eigenvalues[i, :] = eigenvalues[::-1]
            #all_eigenvectors[i, :, :] = eigenvectors[:, ::-1]
            all_eigenvalues[i, :], all_eigenvectors[i, :, :] = PCA(cloud_points[indices, :])

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius, epsilon=1e-6):
    """
    Compute 4 features for all query points using local PCA.
    
    Parameters:
    - query_points: (N, 3) numpy array representing the query points.
    - cloud_points: (M, 3) numpy array representing the cloud points.
    - radius: Neighborhood radius for PCA computation.
    - epsilon: Small value to avoid division by zero.
    
    Returns:
    - features: (N, 4) numpy array with computed features.
    """
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    
    # Sort eigenvalues in descending order
    λ1, λ2, λ3 = all_eigenvalues[:, 2], all_eigenvalues[:, 1], all_eigenvalues[:, 0]

    # Compute features
    linearity = (λ1 - λ2) / (λ1 + epsilon)
    planarity = (λ2 - λ3) / (λ1 + epsilon)
    sphericity = λ3 / (λ1 + epsilon)

    # Compute verticality
    e3 = all_eigenvectors[:, :, 0]  # Normal vector is the first eigenvector (smallest eigenvalue)
    verticality = 2 * np.arcsin(np.abs(e3[:, 2])) / np.pi  # e3[:, 2] is the Z component of the normal

    # Stack features into a (N, 4) array
    features = np.column_stack((linearity, planarity, sphericity, verticality))

    print(features)
    return features


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    if True:
        
        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        
        # Compute features
        features = compute_features(cloud, cloud, 0.50)
        
        # Save cloud with features
        write_ply('../Lille_street_small_features.ply', (cloud, features), ['x', 'y', 'z', 'linearity', 'planarity', 'sphericity', 'verticality'])
        