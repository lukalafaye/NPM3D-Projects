#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # YOUR CODE
    neighborhoods = []
    radius_squared = radius ** 2

    for query in queries:
        distances_squared = np.sum((supports - query) ** 2, axis=1)
        neighbors = np.where(distances_squared < radius_squared)[0]
        neighborhoods.append(neighbors)

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = []

    for query in queries:
        distances_squared = np.sum((supports - query) ** 2, axis=1)
        neighbors = np.argpartition(distances_squared, k)[:k]
        neighborhoods.append(neighbors)

    return neighborhoods





# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000
        radius = 0.2
        leaf_sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 100, 1000]
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        times = []
        for leaf_size in leaf_sizes:
            tree = KDTree(points, leaf_size=leaf_size)

            t0 = time.time()
            tree.query_radius(queries, r=radius)
            t1 = time.time()

            query_time = t1 - t0
            times.append(query_time)
            print(f'Leaf size {leaf_size}: {query_time:.3f} seconds for {num_queries} queries')

        optimal_leaf_size = leaf_sizes[np.argmin(times)]
        print(f'Optimal leaf size: {optimal_leaf_size}')

        tree = KDTree(points, leaf_size=optimal_leaf_size)
        t0 = time.time()
        tree.query_radius(queries, r=radius)
        t1 = time.time()

        print(f'Time with optimal leaf size ({optimal_leaf_size}): {t1 - t0:.3f} seconds for {num_queries} queries')        
        

        radii = [0.1, 0.2, 0.5, 1.0, 2.0]

        optimal_leaf_size = 25
        tree = KDTree(points, leaf_size=optimal_leaf_size)

        timings = []

        for radius in radii:
            t0 = time.time()
            tree.query_radius(queries, r=radius)
            t1 = time.time()
            timings.append(t1 - t0)
            print(f'Radius {radius:.1f}: {t1 - t0:.3f} seconds for {num_queries} queries')

        plt.figure()
        plt.plot(radii, timings, marker='o')
        plt.title('Timing of KDTree query as a function of radius')
        plt.xlabel('Radius (m)')
        plt.ylabel('Query Time (seconds)')
        plt.grid(True)
        plt.savefig('timing_vs_radius.png')

        radius = 0.2
        t0 = time.time()
        tree.query_radius(queries, r=radius)
        t1 = time.time()
        time_per_query = (t1 - t0) / num_queries
        total_time = time_per_query * points.shape[0]
        print(f'Estimated time to compute 20cm neighborhoods for the whole cloud: {total_time / 3600:.2f} hours')

            
            