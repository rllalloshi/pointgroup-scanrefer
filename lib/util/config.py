'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os

from lib.config import CONF

path = os.path.join(CONF.PATH.BASE, 'config/pointgroup_run1_scannet.yaml')
def get_parser():
    args = argparse.ArgumentParser(description='Point Cloud Segmentation')
    with open(path, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    return args


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, path.split('/')[-1][:-5]))
