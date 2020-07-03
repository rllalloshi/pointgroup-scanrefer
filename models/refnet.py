import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.ref_module import RefModule
from lib.pointgroup import PointGroup
from lib.util.config import cfg
from lib.pointgroup import model_fn_decorator
class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, input_feature_dim=0, num_proposal=256, vote_factor=1, sampling="vote_fps", use_lang_classifier=True):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling
        self.use_lang_classifier=use_lang_classifier

        # Backbone point feature learning
        self.backbone_net = PointGroup(cfg)

        # Vote aggregation, detection and language reference
        self.rfnet = RefModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, use_lang_classifier)

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        model_fn = model_fn_decorator()
        ret = model_fn(data_dict, self.backbone_net, 1)

        scores, score_feats,  proposals_idx, proposals_offset = ret['proposal_scores']
        ious = ret['ious']
        batch_offsets = ret["batch_offsets"]
        batch_instance_offsets = data_dict["batch_instance_offsets"]
        batch_size = ret["batch_offsets"].shape[0] - 1


        scores = scores.cuda()
        score_feats = score_feats.cuda()
        proposals_idx = proposals_idx.cuda()

        batch_offsets = batch_offsets.cpu().numpy()
        batch_indices = {}

        N = data_dict['feats'].shape[0]
        scores_pred = torch.sigmoid(scores.view(-1))
        proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int,
                                     device=scores_pred.device)  # (nProposal, N), int, cuda
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

        proposals_to_keep = np.arange(score_feats.shape[0])
        '''
        ##### score threshold
        score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]
        score_feats = score_feats[score_mask]
        proposals_to_keep = score_mask.nonzero().squeeze()

        ##### npoint threshold
        proposals_pointnum = proposals_pred.sum(1)
        npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
        scores_pred = scores_pred[npoint_mask]
        score_feats = score_feats[npoint_mask]
        proposals_to_keep = proposals_to_keep[npoint_mask.nonzero()].squeeze()

        ##### nms
        proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
        intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
        proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
        proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
        proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
        cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
        pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(),
                                            cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
        pick_idxs.sort()
       
        score_feats = score_feats[pick_idxs]
        proposals_to_keep = proposals_to_keep[pick_idxs]
         '''

        proposals_to_keep = proposals_to_keep.squeeze()
        proposals_offset = proposals_offset.cpu().numpy()

        for x in range(batch_size):
            batch_indices[x] = []

        batch = torch.zeros(batch_size, self.num_proposal, 16)
        number_of_proposals_in_batch = torch.zeros(batch_size)
        proposal_mask = torch.zeros(batch_size, self.num_proposal)

        proposal_i = -1
        keep_proposal_i = 0
        for x in proposals_offset[:-1]:
            proposal_i = proposal_i + 1
            if proposal_i not in proposals_to_keep:
                # proposal masked out
                print('proposal masked out')
                continue
            point_idx_in_proposal = proposals_idx[x][1]
            batch_idx = None
            for i in range(batch_size):
                if point_idx_in_proposal < batch_offsets[i+1] and point_idx_in_proposal >= batch_offsets[i]:
                    batch_idx = i
                    break
            batch_indices[batch_idx].append(keep_proposal_i)
            keep_proposal_i = keep_proposal_i + 1
        proposal_ious = []
        for x in range(batch_size):
            batch_inds = batch_indices[x]
            batch_ious = ious[batch_inds]
            batch_scores_feats = score_feats[batch_inds]
            batch_scores_pred = scores_pred[batch_inds]
            batch_score_feats = torch.zeros(self.num_proposal, 16).cuda()
            if batch_scores_feats.shape[0] > self.num_proposal:
                _, batch_proposals_indices = torch.topk(batch_scores_pred.squeeze(), k=self.num_proposal)
                batch_scores_feats = batch_scores_feats[batch_proposals_indices, :]
                batch_ious = batch_ious[batch_proposals_indices, :]

            batch_score_feats[0:batch_scores_feats.shape[0]] = batch_scores_feats
            proposal_mask[x, 0:batch_scores_feats.shape[0]] = 1
            batch[x] = batch_score_feats
            proposal_ious.append(batch_ious)

        batch = batch.cuda()
        data_dict['proposal_ious'] = proposal_ious
        data_dict['batch_instance_offsets'] = batch_instance_offsets
        data_dict['proposal_mask'] = proposal_mask
        data_dict['pg_loss'] = ret['pg_loss']
        data_dict['number_of_proposals_in_batch'] = number_of_proposals_in_batch
        data_dict = self.rfnet(data_dict['locs'], batch, data_dict)

        return data_dict

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)
