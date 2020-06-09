import open3d as o3d
import numpy as np

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


def numpy_to_point_cloud(np_array, standardize_colors=True):
    np_array = np_array[:, 0:6]
    if standardize_colors:
        np_array[:, 3:] = (np_array[:, 3:] - MEAN_COLOR_RGB) / 256.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(np_array[:, 3:6])
    return pcd


def visualize_numpy_array(np_array, standardize_colors=True):
    pcd = numpy_to_point_cloud(np_array, standardize_colors)
    o3d.visualization.draw_geometries([pcd])