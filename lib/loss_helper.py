# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, box3d_iou


def compute_reference_loss(data_dict, config, use_lang_classifier=False, use_max_iou=False):
    # unpack predictions
    cluster_preds = data_dict["cluster_ref"]  # (B, num_proposal)

    # unpack true values
    cluster_labels = data_dict["ref_box_label"]  # (B, num_max_obj)

    # reference loss
    REFERENCE_CLS_WEIGHTS = [0.01, 1]  # put larger weights on positive reference
    criterion = SoftmaxRankingLoss(REFERENCE_CLS_WEIGHTS)
    ref_loss = criterion(cluster_preds, cluster_labels.float())

    # language loss
    if use_lang_classifier:
        criterion = torch.nn.CrossEntropyLoss()
        lang_loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])
    else:
        lang_loss = torch.zeros(1)[0].cuda()
    return ref_loss, lang_loss, cluster_preds, cluster_labels


def get_loss(data_dict, config, reference=False, use_lang_classifier=False, use_max_iou=False, post_processing=None):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    # Compute object classification from features loss
    obj_cls_loss_func = nn.CrossEntropyLoss()

    gt_scene_objects_mask = data_dict['gt_scene_objects_mask']
    gt_scene_objects_mask = gt_scene_objects_mask.view(gt_scene_objects_mask.shape[0] * gt_scene_objects_mask.shape[1])
    object_mask = gt_scene_objects_mask.view(gt_scene_objects_mask.shape[0], 1).repeat(1, 18)

    labels = data_dict["sem_cls_label"]
    labels = labels.view(labels.shape[0] * labels.shape[1])

    logits = data_dict["object_classifier"]*object_mask
    obj_cls_loss = obj_cls_loss_func(logits, labels)
    data_dict["obj_cls_loss"] = obj_cls_loss


    # Calculate object classification accuracy
    with torch.no_grad():
        obj_acc = torch.argmax(data_dict["object_classifier"], dim=1) == labels

        # Get indexes of true objects in order to calculate accuracy only in those objects
        true_object_indexes = gt_scene_objects_mask.nonzero()
        true_object_indexes = true_object_indexes.view(true_object_indexes.shape[0]*true_object_indexes.shape[1])
        obj_acc = obj_acc.take(true_object_indexes)
        obj_acc = obj_acc.float().mean()
        data_dict["obj_acc"] = obj_acc

    '''ref_loss, lang_loss, cluster_preds_scores, cluster_labels = compute_reference_loss(data_dict, config,
                                                                                       use_lang_classifier, use_max_iou)
    data_dict["ref_loss"] = ref_loss
    data_dict["lang_loss"] = lang_loss

    gt_scene_objects_mask = data_dict["gt_scene_objects_mask"]
    cluster_preds = torch.argmax(cluster_preds_scores * gt_scene_objects_mask, 1).long().unsqueeze(1).repeat(1, cluster_preds_scores.shape[1])

    TOP_K = 5
    _, cluster_preds_top5 = torch.topk(cluster_preds_scores * gt_scene_objects_mask, TOP_K, 1)
    preds = torch.zeros(gt_scene_objects_mask.shape).cuda()
    preds = preds.scatter_(1, cluster_preds, 1)
    cluster_preds = preds
    cluster_labels = cluster_labels.float()

    # compute classification scores
    corrects = torch.sum((cluster_preds == 1) * (cluster_labels == 1), dim=1).float()
    labels = torch.ones(corrects.shape[0]).cuda()
    ref_acc = corrects / (labels + 1e-8)

    corrects_top5 = torch.zeros_like(corrects)
    for i in range(TOP_K):
        x = cluster_preds_top5[:, i]
        cluster_preds_x = x.long().unsqueeze(1).repeat(1, cluster_preds_scores.shape[1])
        preds_x = torch.zeros(cluster_preds_scores.shape).cuda()
        preds_x = preds_x.scatter_(1, cluster_preds_x, 1)
        cluster_preds = preds_x
        corrects_top5 += torch.sum((cluster_preds == 1) * (cluster_labels == 1), dim=1).float()
        labels = torch.ones(corrects.shape[0]).cuda()
    ref_acc_top5 = corrects_top5 / (labels + 1e-8)

    # store
   # data_dict["ref_acc"] = ref_acc.cpu().numpy().tolist()
   # data_dict["ref_acc_top5"] = ref_acc_top5.cpu().numpy().tolist()
'''

    data_dict["ref_loss"] = torch.tensor(0)
    data_dict["lang_loss"] = torch.tensor(0)
    data_dict["ref_acc"] = np.array([0, 0, 0])
    data_dict["ref_acc_top5"] = np.array([0, 0, 0])

    # compute localization metrics
    # pred_ref = torch.argmax(data_dict['cluster_ref'] * data_dict['gt_scene_objects_mask'], 1).detach().cpu().numpy()  # (B,)
    # gt_ref = torch.argmax(data_dict["ref_box_label"], 1).detach().cpu().numpy()
    # gt_center = data_dict['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    # gt_heading_class = data_dict['heading_class_label'].cpu().numpy()  # B,K2
    # gt_heading_residual = data_dict['heading_residual_label'].cpu().numpy()  # B,K2
    # gt_size_class = data_dict['size_class_label'].cpu().numpy()  # B,K2
    # gt_size_residual = data_dict['size_residual_label'].cpu().numpy()  # B,K2,3

    '''ious = []
    for i in range(pred_ref.shape[0]):
        # compute the iou
        pred_ref_idx, gt_ref_idx = pred_ref[i], gt_ref[i]
        pred_obb = config.param2obb(gt_center[i, pred_ref_idx, 0:3], gt_heading_class[i, pred_ref_idx],
                                    gt_heading_residual[i, pred_ref_idx],
                                    gt_size_class[i, pred_ref_idx], gt_size_residual[i, pred_ref_idx])
        gt_obb = config.param2obb(gt_center[i, gt_ref_idx, 0:3], gt_heading_class[i, gt_ref_idx],
                                  gt_heading_residual[i, gt_ref_idx],
                                  gt_size_class[i, gt_ref_idx], gt_size_residual[i, gt_ref_idx])
        pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
        gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])
        iou, _ = box3d_iou(pred_bbox, gt_bbox)
        ious.append(iou)

    # store
    data_dict["ref_iou"] = ious
    data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
    data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

    loss = ref_loss + 2*lang_loss + 4*obj_cls_loss
    loss *= 10  # amplify
    '''
    loss = obj_cls_loss
    loss *= 10
    data_dict['loss'] = loss
    data_dict["ref_iou_rate_0.25"] = np.array([0, 0, 0])
    data_dict["ref_iou_rate_0.5"] = np.array([0, 0, 0])

    # lang acc
    '''
    if use_lang_classifier:
        data_dict["lang_acc"] = (torch.argmax(data_dict['lang_scores'], 1) == data_dict["object_cat"]).float().mean()
    else:
        data_dict["lang_acc"] = torch.zeros(1)[0].cuda()
    '''
    data_dict["lang_acc"] = torch.zeros(1)[0].cuda()
    return loss, data_dict
