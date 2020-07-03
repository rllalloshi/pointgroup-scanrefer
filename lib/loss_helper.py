import torch
import sys
import os
import numpy as np


sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.loss import SoftmaxRankingLoss

NUM_PROPOSALS = 256 # TODO change this

def compute_reference_loss(data_dict):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)
    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # for each batch i get max iou with gt object and mark that as the correct prediction
    # ious tells us for each proposal iou with gt object, we have object id so we get proposal that has max iou
    # with object id and consider that as our true label
    proposal_ious = data_dict['proposal_ious']
    object_id = data_dict['object_id']
    batch_instance_offsets = data_dict['batch_instance_offsets']

    cluster_preds = data_dict["cluster_ref"].float().cuda().clone() # (B, num_proposal)
    true_labels = torch.zeros(cluster_preds.shape).float().cuda()
    for batch in range(cluster_preds.shape[0]):
        batch_ious = proposal_ious[batch] # batch ious holds proposals - object ious shape (nProposals, nObjects)
        print(f"[BATCH_IOU] {batch_ious.shape}")
        batch_instance_offset = batch_instance_offsets[batch]
        # this is the gt referenced object id, i add batch_instance_offset because of how pointgroup handles batches nObjects is not
        # only object in this scene but all batch scenes, so the second batch item, if gt object is 1, in batch_ious it wont
        # be in batch_ious[proposalnumber][1] but 1 + number of object in first batch item
        gt_object_id = object_id[batch] + batch_instance_offset
        gt_object_ious = batch_ious[:, gt_object_id] # i get all ious for this gt object id
        gt_object_proposal_idx = gt_object_ious.argmax() # what proposal has maximum iou for gt object id
        true_labels[batch][gt_object_proposal_idx] = 1 # this is the true label, the proposal we want to have the hightest prediction

    REFERENCE_CLS_WEIGHTS = [1/NUM_PROPOSALS, 1] # put larger weights on positive reference
    criterion = SoftmaxRankingLoss(REFERENCE_CLS_WEIGHTS)
    ref_loss = criterion(cluster_preds, true_labels)

    return ref_loss, cluster_preds, true_labels

def get_loss(data_dict):
    """ Loss functions

    Args:
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # Reference loss
    ref_loss, cluster_preds, cluster_labels = compute_reference_loss(data_dict)
    data_dict["ref_loss"] = ref_loss

    # Lang loss
    criterion = torch.nn.CrossEntropyLoss()
    lang_loss = criterion(data_dict["lang_scores"].cuda(), data_dict["object_cat"].cuda().view(-1))
    data_dict["lang_loss"] = lang_loss

    loss = 0.1*ref_loss + lang_loss

    loss *= 10 # amplify
    data_dict['loss'] = loss

    with torch.no_grad():
        # reference accuracy
        cluster_preds = torch.argmax(cluster_preds, dim=1).long().unsqueeze(1).repeat(1, cluster_preds.shape[1])
        preds = torch.zeros(cluster_preds.shape).cuda()
        preds = preds.scatter_(1, cluster_preds, 1)
        cluster_preds = preds.cuda()
        corrects = torch.sum((cluster_preds == 1) * (cluster_labels == 1), dim=1).float()
        labels = torch.ones(corrects.shape[0]).cuda()
        ref_acc = (corrects / (labels + 1e-8)).cpu().numpy()

        object_cat = data_dict["object_cat"].cuda()

        data_dict["lang_acc"] = (torch.argmax(data_dict['lang_scores'], 1) == object_cat).float().mean()
        data_dict["ref_acc"] = ref_acc

        # ious
        ious = []
        proposal_ious = data_dict['proposal_ious']
        proposal_mask = data_dict['proposal_mask'].float().cuda()
        object_id = data_dict['object_id']
        cluster_ref = data_dict['cluster_ref']

        predictions = torch.argmax(cluster_ref*proposal_mask, 1).detach().cpu().numpy()
        batch_instance_offsets = data_dict['batch_instance_offsets']

        print(f"[1]cluster_ref: {cluster_ref.shape}")

        print(f"[3]proposal_mask: {proposal_mask.shape}")
        print(f"[6]predictions: {predictions}")


        for batch in range(cluster_ref.shape[0]):
            gt_object_id = object_id[batch] + batch_instance_offsets[batch]
            print(f"gt_object_id {gt_object_id}; object_id[batch] {object_id[batch]} ; batch_instance_offsets[batch] {batch_instance_offsets[batch]}")
            batch_proposals = proposal_ious[batch]
            print(f"[BATCH_IOU_EVAL]batch_proposals: {batch_proposals.shape}")
            if predictions[batch] >= len(batch_proposals):
                print('problem')
            prediction_ious = batch_proposals[predictions[batch]]
            ious.append(prediction_ious[gt_object_id])

        print(ious)
        data_dict["ref_iou"] = ious
        data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
        data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]



    return loss, data_dict
