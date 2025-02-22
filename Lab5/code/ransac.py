#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from descriptors import compute_local_PCA  

import os



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def compute_plane(points):
    """ Compute the plane passing through 3 points."""
    p0, p1, p2 = points[0], points[1], points[2]

    v1 = p1 - p0
    v2 = p2 - p0

    normal_plane = np.cross(v1, v2)

    normal_plane = normal_plane / np.linalg.norm(normal_plane)

    return p0, normal_plane

def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):
    """ Check if points are in the plane based on distance."""
    distances = np.abs(np.dot(points - pt_plane.T, normal_plane).flatten())
    
    indexes = distances < threshold_in
    
    return indexes

def RANSAC(points, nb_draws=100, threshold_in=0.1):
    """ RANSAC plane fitting."""
    best_vote = 0
    best_pt_plane = None
    best_normal_plane = None

    for _ in range(nb_draws):
        sample_points = points[np.random.choice(points.shape[0], 3, replace=False)]
        
        # Compute the plane
        pt_plane, normal_plane = compute_plane(sample_points)
        
        # Count inliers
        inliers = in_plane(points, pt_plane, normal_plane, threshold_in)
        vote = np.sum(inliers)

        # Update best plane if this one has more inliers
        if vote > best_vote:
            best_vote = vote
            best_pt_plane = pt_plane
            best_normal_plane = normal_plane

    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):
    """ Recursive RANSAC to extract multiple planes."""
    
    remaining_inds = np.arange(len(points))
    plane_inds = []
    plane_labels = np.zeros(len(points), dtype=int)

    for i in range(1, nb_planes + 1):
        if len(remaining_inds) < 3:
            break

        best_pt_plane, best_normal_plane, _ = RANSAC(points[remaining_inds], nb_draws, threshold_in)

        # Find inliers
        inliers = in_plane(points[remaining_inds], best_pt_plane, best_normal_plane, threshold_in)

        # Store indices
        plane_inds.append(remaining_inds[inliers])
        plane_labels[remaining_inds[inliers]] = i

        # Update remaining points
        remaining_inds = remaining_inds[~inliers]

    return plane_inds, remaining_inds, plane_labels


def in_plane_with_normals(points, normals, pt_plane, normal_plane, threshold_in=0.1, normal_thresh=0.9):
    """ Check if points are in the plane based on distance and normal consistency."""
    distances = np.abs(np.dot(points - pt_plane.T, normal_plane).flatten())
    normal_similarities = np.abs(np.dot(normals, normal_plane).flatten())
    
    indexes = (distances < threshold_in) & (normal_similarities > normal_thresh)
    return indexes

def RANSAC_with_normals(points, normals, nb_draws=100, threshold_in=0.1, normal_thresh=0.9):
    """ RANSAC plane fitting considering normal vectors."""
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    for _ in range(nb_draws):
        pts = points[np.random.randint(0, len(points), size=3)]
        pt_plane, normal_plane = compute_plane(pts)
        
        inliers = in_plane_with_normals(points, normals, pt_plane, normal_plane, threshold_in, normal_thresh)
        vote = inliers.sum()
        
        if vote > best_vote:
            best_vote = vote
            best_pt_plane = pt_plane
            best_normal_plane = normal_plane
    
    return best_pt_plane, best_normal_plane, best_vote

def recursive_RANSAC_with_normals(points, normals, nb_draws=100, threshold_in=0.1, normal_thresh=0.9, nb_planes=5):
    """ Recursive RANSAC to extract multiple planes using normal constraints."""
    nb_points = len(points)
    plane_inds = []
    plane_labels = np.zeros(nb_points, dtype=int)
    remaining_inds = np.arange(nb_points)
    
    for plane_id in range(nb_planes):
        if len(remaining_inds) < 3:
            break 
        
        best_pt_plane, best_normal_plane, best_vote = RANSAC_with_normals(points[remaining_inds], normals[remaining_inds], nb_draws, threshold_in, normal_thresh)
        inliers = in_plane_with_normals(points[remaining_inds], normals[remaining_inds], best_pt_plane, best_normal_plane, threshold_in, normal_thresh)
        
        if inliers.sum() == 0:
            break 
        
        plane_inds.append(remaining_inds[inliers])
        plane_labels[remaining_inds[inliers]] = plane_id + 1  # Assign plane ID (starting from 1)
        remaining_inds = remaining_inds[~inliers]
    
    return plane_inds, remaining_inds, plane_labels

########## BONUS

from joblib import Parallel, delayed

def accelerated_RANSAC_batch_recursive(points,
                                       normals,
                                       nb_draws=500,
                                       batch_size=50,
                                       threshold_in=0.1,
                                       normal_thresh=0.9,
                                       nb_planes=5):

    def RANSAC_with_normals_parallel(pts, pts_normals, nb_draws, threshold_in, normal_thresh):

        # pregenerate random triplets in one go to reduce overhead
        rng = np.random.default_rng()
        triplets = [rng.choice(len(pts), 3, replace=False) for _ in range(nb_draws)]

        def single_ransac_iteration(sample_idx):
            sample_points = pts[sample_idx]
            pt_plane, normal_plane = compute_plane(sample_points)

            inliers_mask = in_plane_with_normals(pts, pts_normals,
                                                 pt_plane, normal_plane,
                                                 threshold_in, normal_thresh)

            vote = np.sum(inliers_mask)
            return pt_plane, normal_plane, vote

        results = Parallel(n_jobs=-1)(
            delayed(single_ransac_iteration)(triplets[i]) for i in range(nb_draws)
        )

        best_pt_plane, best_normal_plane, best_vote = max(results, key=lambda x: x[2])
        return best_pt_plane, best_normal_plane, best_vote

    plane_inds_list = []
    plane_labels = np.zeros(len(points), dtype=int)
    remaining_inds = np.arange(len(points))

    for plane_id in range(nb_planes):

        if len(remaining_inds) < 3:
            break

        sub_points = points[remaining_inds]
        sub_normals = normals[remaining_inds]

        best_pt_plane, best_normal_plane, best_vote = RANSAC_with_normals_parallel(
            sub_points, sub_normals, nb_draws, threshold_in, normal_thresh
        )

        inliers_mask = in_plane_with_normals(sub_points, sub_normals,
                                             best_pt_plane, best_normal_plane,
                                             threshold_in, normal_thresh)

        if np.sum(inliers_mask) == 0:
            break

        current_plane_inds = remaining_inds[inliers_mask]
        plane_inds_list.append(current_plane_inds)
        plane_labels[current_plane_inds] = plane_id + 1

        remaining_inds = remaining_inds[~inliers_mask]

    return plane_inds_list, remaining_inds, plane_labels


def load_normals(points, radius=0.5, file_path='../data/normals.npy'):
    if os.path.exists(file_path):
        t0 = time.time()
        normals = np.load(file_path)
        t1 = time.time()
        print(f'Loaded normals in {t1 - t0:.3f} seconds from {file_path}')
    else:
        print('\n--- Computing Normals using PCA ---')
        t0 = time.time()
        # Compute normals using PCA; the smallest eigenvector is taken as the normal.
        _, eigenvectors = compute_local_PCA(points, points, radius=radius)
        normals = eigenvectors[:, :, 0]
        t1 = time.time()
        print(f'Normal computation done in {t1 - t0:.3f} seconds')
        np.save(file_path, normals)
    return normals


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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    
    use_normals = False  # Set to True to use the method with normals
    use_Lille = False  # Set to True to use the Lille dataset
    bonus = True  # Set to True to use the bonus method
    
    if bonus:
        normals = load_normals(points, radius=0.5, file_path='../data/normals.npy')
        print('\n--- Bonus ---\n')
        
        print("\n--- Running recursive RANSAC with Normals  ---")
        nb_draws = 100
        threshold_in = 0.15
        normal_thresh = 0.6
        nb_planes = 5

        t0 = time.time()
        plane_inds_norm_list, remaining_inds_norm, plane_labels_norm = recursive_RANSAC_with_normals(
            points, normals, nb_draws, threshold_in, normal_thresh, nb_planes
        )
        t1 = time.time()
        normal_ransac_time = t1 - t0
        print(f'BONUS recursive RANSAC with Normals RANSAC completed in {normal_ransac_time:.3f} seconds')

        plane_inds_norm = np.concatenate(plane_inds_norm_list) if len(plane_inds_norm_list) > 0 else np.array([], dtype=int)
        plane_labels_norm_sub = plane_labels_norm[plane_inds_norm] if len(plane_inds_norm) > 0 else np.array([], dtype=int)

        write_ply('../normal_ransac_planes.ply',
                [points[plane_inds_norm], plane_labels_norm_sub.astype(np.int32)],
                ['x', 'y', 'z', 'plane_label'])
        
        write_ply('../normal_ransac_remaining.ply',
                [points[remaining_inds_norm]], ['x', 'y', 'z'])

        print("\n--- Running Bonus Accelerated RANSAC ---")
        
        
        t0 = time.time()
        plane_inds_acc_list, remaining_inds_acc, plane_labels_acc = accelerated_RANSAC_batch_recursive(
            points,
            normals,
            nb_draws=nb_draws,
            batch_size=50,
            threshold_in=threshold_in,
            normal_thresh=normal_thresh,
            nb_planes=nb_planes
        )

        t1 = time.time()
        accelerated_ransac_time = t1 - t0
        print(f'Bonus Accelerated RANSAC completed in {accelerated_ransac_time:.3f} seconds')


        plane_inds_acc = np.concatenate(plane_inds_acc_list) if len(plane_inds_acc_list) > 0 else np.array([], dtype=int)
        plane_labels_acc_sub = plane_labels_acc[plane_inds_acc] if len(plane_inds_acc) > 0 else np.array([], dtype=int)

        # Extract colors corresponding to the detected plane points
        plane_colors_norm = colors[plane_inds_norm] if len(plane_inds_norm) > 0 else np.array([], dtype=int)
        plane_colors_acc = colors[plane_inds_acc] if len(plane_inds_acc) > 0 else np.array([], dtype=int)

        # Save the normal-based RANSAC planes with colors
        write_ply('../normal_ransac_planes.ply',
                [points[plane_inds_norm], plane_colors_norm, plane_labels_norm_sub.astype(np.int32)],
                ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])

        # Save the accelerated RANSAC planes with colors
        write_ply('../accelerated_ransac_planes.ply',
                [points[plane_inds_acc], plane_colors_acc, plane_labels_acc_sub.astype(np.int32)],
                ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])


        # Compare times
        print("\n--- Comparison Results ---")
        print(f'Normal RANSAC Time: {normal_ransac_time:.3f} seconds')
        print(f'Accelerated RANSAC Time: {accelerated_ransac_time:.3f} seconds')
        speedup = normal_ransac_time / accelerated_ransac_time if accelerated_ransac_time > 0 else float('inf')
        print(f'Acceleration Speedup: {speedup:.2f}x')

        print('Done!')
        exit(1)
    if not use_normals and not use_Lille:
        # Computes the plane passing through 3 randomly chosen points
        # ************************
        #
        
        print('\n--- 1) and 2) ---\n')
        
        # Define parameter
        threshold_in = 0.10

        # Take randomly three points
        pts = points[np.random.randint(0, nb_points, size=3)]
        
        # Computes the plane passing through the 3 points
        t0 = time.time()
        pt_plane, normal_plane = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))
        
        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]
        
        # Save extracted plane and remaining points
        write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        
        # Computes the best plane fitting the point cloud
        # ***********************************
        #
        #
        
        print('\n--- 3) ---\n')

        # Define parameters of RANSAC
        nb_draws = 100
        threshold_in = 0.10

        # Find best plane by RANSAC
        t0 = time.time()
        best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
        
        # Find points in the plane and others
        points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]
        
        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        

        # Find "all planes" in the cloud
        # ***********************************
        #
        #
        
        print('\n--- 4) ---\n')
        
        # Define parameters of recursive_RANSAC
        nb_draws = 100
        threshold_in = 0.10
        nb_planes = 2
        
        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
        t1 = time.time()
        print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
        
        # Ensure all extracted plane indices are properly concatenated
        if len(plane_inds) > 0:
            plane_inds = np.concatenate(plane_inds)
            plane_labels = plane_labels[plane_inds]  # Match labels to indices
        else:
            plane_inds = np.array([], dtype=int)  # Ensure it's a valid empty array
            plane_labels = np.array([], dtype=int)  # Ensure labels array is also empty

                    
        # Save the best planes and remaining points
        write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
        write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    elif use_normals:
        print('\n--- Computing Normals using PCA from TP3 ---\n')
        t0 = time.time()
        _, eigenvectors = compute_local_PCA(points, points, radius=0.5)  # Use PCA for normal estimation
        normals = eigenvectors[:, :, 0]  # Smallest eigenvector is the normal
        t1 = time.time()
        print(f'Normal computation done in {t1 - t0:.3f} seconds')
        
        print('\n--- Running RANSAC with Normals ---\n')
        nb_draws = 100
        threshold_in = 0.10
        normal_thresh = 0.5
        nb_planes = 5
        
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC_with_normals(points, normals, nb_draws, threshold_in, normal_thresh, nb_planes)
        t1 = time.time()
        print(f'Recursive RANSAC done in {t1 - t0:.3f} seconds')
        
        plane_inds = np.concatenate(plane_inds) if len(plane_inds) > 0 else np.array([], dtype=int)
        plane_labels = plane_labels[plane_inds] if len(plane_inds) > 0 else np.array([], dtype=int)
        
        write_ply('../segmented_planes.ply', [points[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'plane_label'])
        write_ply('../remaining_points.ply', [points[remaining_inds]], ['x', 'y', 'z'])
    
    elif use_Lille:
        file_path = '../data/Lille_street_small.ply'

        # Load point cloud
        data = read_ply(file_path)

        # Concatenate data
        points = np.vstack((data['x'], data['y'], data['z'])).T
        nb_points = len(points)
        
        print('\n--- Computing RANSAC for Lille_street_small plot ---\n')
        
        t0 = time.time()
        threshold_in = 0.3 
        
        pts = points[np.random.randint(0, nb_points, size=3)]
        pt_plane, normal_plane = compute_plane(pts)
        
        points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
        
        plane_inds = points_in_plane.nonzero()[0]
        
        remaining_inds = (1-points_in_plane).nonzero()[0]
        
        write_ply('../plane_Lille.ply', [points[plane_inds], labels[plane_inds]], ['x', 'y', 'z'])
        
        threshold_in = 0.1
        nb_draws = 100
        
        best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
        t1 = time.time()
        
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
        
        points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
        
        plane_inds = points_in_plane.nonzero()[0]
        
        remaining_inds = (1-points_in_plane).nonzero()[0]
        
        write_ply('../best_plane_Lille.ply', [points[plane_inds], labels[plane_inds]], ['x', 'y', 'z'])
        
        threshold_in = 0.1
        nb_draws = 100
        
        nb_planes = 2
        
        plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
        
        plane_inds = np.concatenate(plane_inds)
        
        plane_labels = plane_labels[plane_inds]
        
        write_ply('../best_planes_Lille.ply', [points[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'label', 'plane_label'])

    print('Done!')

    