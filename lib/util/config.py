'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os

def get_parser():
    args = argparse.ArgumentParser(description='Point Cloud Segmentation')
    with open('/datadrive/forked/NEW/config/pointgroup_run1_scannet.yaml', 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    return args


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, '/datadrive/forked/NEW/config/pointgroup_run1_scannet.yaml'.split('/')[-1][:-5]))
