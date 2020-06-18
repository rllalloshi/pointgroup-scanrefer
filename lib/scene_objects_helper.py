import numpy as np
import os
from lib.config import CONF
# from lib.o3d_helper import visualize_numpy_array
#
#
# def visualize_scene_objects(scene_id):
#     scene_objects = read_scene_objects(scene_id)
#     for scene_object in scene_objects:
#         visualize_numpy_array(scene_object)


def read_scene_objects(scene_id):
    scene_path = os.path.join(CONF.PATH.SCANNET_DATA, scene_id)
    has_more_objects = True
    object_number = 0
    scene_objects = []
    while has_more_objects:
        try:
            scene_object = np.load(
                scene_path + "_object_" + str(object_number) + ".npy")
            scene_objects.append(scene_object)
            object_number = object_number + 1
        except IOError:
            has_more_objects = False
    return scene_objects
