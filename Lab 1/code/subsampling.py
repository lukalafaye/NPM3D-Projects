#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels



def grid_subsampling(points, colors, labels, voxel_size=0.1):
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    voxel_dict = {}
    for i, voxel_index in enumerate(map(tuple, voxel_indices)):
        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = {"points": [], "colors": [], "labels": []}
        voxel_dict[voxel_index]["points"].append(points[i])
        voxel_dict[voxel_index]["colors"].append(colors[i])
        voxel_dict[voxel_index]["labels"].append(labels[i])

    subsampled_points = []
    subsampled_colors = []
    subsampled_labels = []

    for voxel_data in voxel_dict.values():
        subsampled_points.append(np.mean(voxel_data["points"], axis=0))
        subsampled_colors.append(np.mean(voxel_data["colors"], axis=0))
        subsampled_labels.append(np.bincount(voxel_data["labels"]).argmax())

    subsampled_points = np.array(subsampled_points)
    subsampled_colors = np.array(subsampled_colors)
    subsampled_labels = np.array(subsampled_labels)

    return subsampled_points, subsampled_colors, subsampled_labels

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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Grid subsampling
    voxel_size = 0.1
    t0 = time.time()

    subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling(
        points, colors, labels, voxel_size
    )

    subsampled_points = subsampled_points.astype(np.float32)
    subsampled_colors = subsampled_colors.astype(np.uint8)
    subsampled_labels = subsampled_labels.astype(np.int32)
    
    t1 = time.time()

    print('grid subsampling done in {:.3f} seconds'.format(t1 - t0))

    write_ply('../grid_decimated.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')
