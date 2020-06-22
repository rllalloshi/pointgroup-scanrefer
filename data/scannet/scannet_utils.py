""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/scannet_utils.py
"""

import os
import sys
import csv

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']

from lib.config import CONF

def represents_int(s):
    ''' if string s represents an int. '''
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def get_raw2scannetv2_label_map():
    path = os.path.join(CONF.PATH.SCANNET, 'meta_data/scannetv2-labels.combined.tsv')
    lines = [line.rstrip() for line in open(path)]
    lines_0 = lines[0].split('\t')
    print(lines_0)
    print(len(lines))
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        if (elements[1] != elements[2]):
            print('{}: {} {}'.format(i, elements[1], elements[2]))
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

g_raw2scannetv2 = get_raw2scannetv2_label_map()

