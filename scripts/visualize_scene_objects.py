import argparse

from lib.scene_objects_helper import visualize_scene_objects

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, help="Scene id for the scene whose object you want to visualize",
                        required=True)
    args = parser.parse_args()

    visualize_scene_objects(args.scene_id)
