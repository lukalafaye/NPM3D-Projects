#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np
import time

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys

#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE

    # Number of points
    num_points = data.shape[1]

    # Step 1: Compute centroids
    pm_moving = (1 / num_points) * np.sum(data, axis=1, keepdims=True)  # (d x 1)
    pm_ref = (1 / num_points) * np.sum(ref, axis=1, keepdims=True)  # (d x 1)

    # Step 2: Center the point clouds
    Q_moving = data - pm_moving  # (d x N)
    Q_ref = ref - pm_ref  # (d x N)

    # Step 3: Compute cross-covariance matrix
    H = np.dot(Q_moving, Q_ref.T)  # (d x d)

    # Step 4: Compute SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=True)

    # Step 5: Check and correct reflection if needed
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0:
        U[:, -1] *= -1  # Reflect the last column of U

    R = np.dot(Vt.T, U.T)

    # Step 7: Compute translation vector
    T = pm_ref - np.dot(R, pm_moving)

    return R, T

def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''


    data_aligned = np.copy(data)
    closest_points = np.zeros_like(data_aligned)

    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    rms = RMS_threshold + 1
    N_data = data.shape[1]
    d = data.shape[0]

    R_cumulative = np.eye(data.shape[0])
    T_cumulative = np.zeros((data.shape[0], 1))

    # Build KDTree for reference cloud
    kd_tree = KDTree(ref.T)

    iter = 0
    while iter < max_iter and rms > RMS_threshold:
        iter += 1

        # Find the nearest neighbors using KDTree
        distances, indices = kd_tree.query(data_aligned.T, k=1)
        neighbors_list.append(indices.flatten())

        # Get the corresponding closest points
        closest_points = ref[:, indices.flatten()]

        # Compute the best rigid transformation
        R, T = best_rigid_transform(data_aligned, closest_points)

        # Update cumulative transformations
        R_cumulative = R @ R_cumulative
        T_cumulative = R @ T_cumulative + T

        R_list.append(R_cumulative)
        T_list.append(T_cumulative)

        # Compute RMS between data_aligned and closest_points
        rms = np.sqrt(np.mean(np.linalg.norm(data_aligned - closest_points, axis=0)**2))
        RMS_list.append(rms)

        print(f"Iteration {iter}: RMS = {rms:.6f}")

        # Apply the transformation
        data_aligned = R @ data_aligned + T

    return data_aligned, R_list, T_list, neighbors_list, RMS_list
 


def icp_point_to_point_fast(data, ref, max_iter=50, RMS_threshold=1e-6, sampling_limit=1000):
    '''
    Iterative closest point algorithm with a point-to-point strategy, using random sampling
    to speed up computation.

    Inputs:
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        sampling_limit = number of points to use at each iteration (randomly sampled)

    Returns:
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each sampled data point in
        the ref cloud and obtain a (1 x sampled_size) array of indices. This is the list of those arrays.
        RMS_list = list of RMS values at each iteration
    '''
    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # Total number of points in the data
    N_data = data.shape[1]

    # KDTree for reference cloud
    tree = KDTree(ref.T)

    for iteration in range(max_iter):
        # Randomly sample points
        sampled_indices = np.random.choice(N_data, size=min(sampling_limit, N_data), replace=False)
        sampled_data = data_aligned[:, sampled_indices]

        # Find closest points in the reference cloud
        distances, indices = tree.query(sampled_data.T, k=1)
        indices = indices.flatten()
        neighbors_list.append(indices)

        # Get the corresponding closest points from the reference cloud
        ref_corresponding = ref[:, indices]

        # Compute the best rigid transformation using sampled points
        R, T = best_rigid_transform(sampled_data, ref_corresponding)

        # Apply the transformation to the entire dataset
        data_aligned = R @ data_aligned + T

        # Store transformations
        R_list.append(R)
        T_list.append(T)

        # Compute RMS error for the sampled points
        RMS = np.sqrt(np.mean(distances**2))
        RMS_list.append(RMS)

        print(f"Iteration {iteration + 1}: RMS = {RMS:.6f}")

        # Stop if RMS is below threshold
        if RMS < RMS_threshold:
            break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.title("Root Mean Square (RMS) Plot for 2D ICP")
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.title("Root Mean Square (RMS) Plot for 3D ICP")
        plt.plot(RMS_list)
        plt.show()


    if False:
        # Cloud paths
        Notre_Dame_o_path = '../data/Notre_Dame_Des_Champs_1.ply'
        Notre_Dame_p_path = '../data/Notre_Dame_Des_Champs_2.ply'
        
        # Load clouds
        Notre_Dame_o_ply = read_ply(Notre_Dame_o_path)
        Notre_Dame_p_ply = read_ply(Notre_Dame_p_path)
        Notre_Dame_o = np.vstack((Notre_Dame_o_ply['x'], Notre_Dame_o_ply['y'], Notre_Dame_o_ply['z']))
        Notre_Dame_p = np.vstack((Notre_Dame_p_ply['x'], Notre_Dame_p_ply['y'], Notre_Dame_p_ply['z']))
        
        # Apply ICP
        Notre_Dame_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(Notre_Dame_p, Notre_Dame_o, 25, 1e-4, 1000)
        Notre_Dame_p_opt2, R_list2, T_list2, neighbors_list2, RMS_list2 = icp_point_to_point_fast(Notre_Dame_p, Notre_Dame_o, 25, 1e-4, 10000)
        
        write_ply('../Notre_Dame_Des_Champs_3.ply', [Notre_Dame_p_opt.T], ['x', 'y', 'z'])

        # Show ICP
        #show_ICP(Notre_Dame_p, Notre_Dame_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.title("Root Mean Square (RMS) Plot for 3D Fast ICP 1k points")
        plt.plot(RMS_list)
        plt.show()

        plt.title("Root Mean Square (RMS) Plot for 3D Fast ICP 100k points")
        plt.plot(RMS_list2)
        plt.show()



                                 
                       