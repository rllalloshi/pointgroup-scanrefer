""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py

Batch mode in loading Scannet scenes with vertices and ground truth labels for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""

import os
import sys
import datetime
import torch
import numpy as np
from load_scannet_data import export

import pdb

SCANNET_DIR = 'scans'
SCAN_NAMES = sorted([line.rstrip() for line in open('meta_data/scannetv2.txt')])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
OUTPUT_FOLDER = './scannet_data'
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'


def export_one_scan(scan_name, output_filename_prefix):
    labels_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.labels.ply')
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    coords, colors, sem_labels, instance_labels, instance_bboxes = export(scan_name, mesh_file, agg_file, seg_file, labels_file,LABEL_MAP_FILE, None)

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-2], OBJ_CLASS_IDS) # match the mesh2cap
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    print("Shape of points: {}".format(coords.shape))

    torch.save((coords, colors, sem_labels, instance_labels), output_filename_prefix + '_inst_nostuff.pth')
    np.save(output_filename_prefix + '_bbox.npy', instance_bboxes)

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
        
    for scan_name in SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)      
        export_one_scan(scan_name, output_filename_prefix)
             
        print('-'*20+'done')

if __name__=='__main__':    
    batch_export()
