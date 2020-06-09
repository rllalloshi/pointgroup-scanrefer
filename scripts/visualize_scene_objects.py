import argparse
import numpy as np
import os

from lib.config import CONF
from lib.o3d_helper import visualize_numpy_array


def visualize_scene_objects(scene_id):
    scene_path = os.path.join(CONF.PATH.SCANNET_DATA, scene_id)
    has_more_objects = True
    object_number = 0
    while has_more_objects:
        try:
            scene_object = np.load(
                scene_path + "_object_" + str(object_number) + ".npy")
            visualize_numpy_array(scene_object)
            object_number = object_number + 1
        except IOError:
            has_more_objects = False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, help="Scene id for the scene whose object you want to visualize",
                        required=True)
    args = parser.parse_args()

    visualize_scene_objects(args.scene_id)
