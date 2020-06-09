import argparse
import numpy as np
import os
import open3d as o3d

from lib.config import CONF


def numpy_to_point_cloud(scene_object):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_object[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(scene_object[:, 3:6])
    return pcd


def visualize_scene_objects(scene_id):
    print(scene_id)
    scene_objects = np.load(
        os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_objects.npy", allow_pickle=True)
    print(scene_objects.shape)
    num_objects = scene_objects.shape[0]
    for i in range(num_objects):
        pcd = numpy_to_point_cloud(scene_objects[i])
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, help="Scene id for the scene whose object you want to visualize",
                        required=True)
    args = parser.parse_args()

    visualize_scene_objects(args.scene_id)
