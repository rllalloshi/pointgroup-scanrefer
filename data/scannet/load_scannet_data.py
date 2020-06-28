""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py

Load Scannet scenes with vertices and ground truth labels for semantic and instance segmentations
"""

# python imports
import os
import scannet_utils
import plyfile, numpy as np, torch, json, argparse


remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

def export(scene_name, mesh_file, agg_file, seg_file, labels_file, label_map_file, output_file=None):
    mesh = plyfile.PlyData().read(mesh_file)
    points = np.array([list(x) for x in mesh.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(labels_file)
    sem_labels = remapper[np.array(f2.elements[0]['label'])]

    with open(seg_file) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    # For each segment point indices
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    # instance segments, array of arrays
    instance_segids = []
    # instance segments labels
    labels = []
    with open(agg_file) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            #if scannet_utils.g_raw2scannetv2[x['label']] != 'wall' and scannet_utils.g_raw2scannetv2[x['label']] != 'floor':
            instance_segids.append(x['segments'])
            labels.append(x['label'])
            assert (x['label'] in scannet_utils.g_raw2scannetv2.keys())

    if (scene_name == 'scene0217_00' and instance_segids[0] == instance_segids[
        int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    # each point what label it belongs to
    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert (len(np.unique(sem_labels[pointids])) == 1)

    if output_file is not None:
        torch.save((coords, colors, sem_labels, instance_labels), output_file + '_inst_nostuff.pth')

    return coords, colors, sem_labels, instance_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')

    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    labels_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.labels.ply')
    export(scan_name, mesh_file, agg_file, seg_file, labels_file, opt.label_map_file, opt.output_file)


if __name__ == '__main__':
    main()
