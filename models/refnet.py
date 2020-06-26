import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.voting_module import VotingModule
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
        batch_offsets = ret["batch_offsets"]
        batch_size = ret["batch_offsets"].shape[0] - 1
        gt_instance_idxs = ret['gt_instance_idxs']

        # # gt_centers = ret["instance_info"][:,0:3]
        # print(f"scores {scores.shape}")
        # print(f"score_feats {score_feats.shape}")
        # print(f"proposals_idx {proposals_idx.shape}")
        # print(f"proposals_offset {proposals_offset.shape}")
        # print(f"batch_offsets {batch_offsets.shape}")
        # # print(f"gt_centers {gt_centers.shape}")
        # print(f"batch_size {batch_size}")

        scores = scores.cuda()
        score_feats = score_feats.cuda()
        proposals_idx = proposals_idx.cuda()
        proposals_offset = proposals_offset.cpu().numpy()
        batch_offsets = batch_offsets.cpu().numpy()
        batch_indices = {}

        for x in range(batch_size):
            batch_indices[x] = []

        batch = torch.zeros(batch_size, self.num_proposal, 16)
        proposal_mask = torch.ones(batch_size, self.num_proposal)
        gt_proposals = torch.ones(batch_size, self.num_proposal)*-1

        for x in proposals_offset[:-1]:
            point_idx_in_proposal = proposals_idx[x][1]
            batch_idx = None
            for i in range(batch_size):
                if point_idx_in_proposal < batch_offsets[i+1] and point_idx_in_proposal >= batch_offsets[i]:
                    batch_idx = i
                    break
            if batch_idx is not None:
                cluster_idx = proposals_idx[x][0].int().item()
                batch_indices[batch_idx].append(cluster_idx)
            else:
                print('No proposal found for instance')

        for x in range(batch_size):
            batch_inds = batch_indices[x]
            batch_scores = scores[batch_inds]
            gt_proposals[x, 0:len(batch_inds)] = gt_instance_idxs[batch_inds]
            # print(f"batch_indx {batch_inds}")

            batch_score_feats = score_feats[batch_inds, :]
            number_of_object_proposals_in_batch = batch_score_feats.shape[0]

            if number_of_object_proposals_in_batch > self.num_proposal:
                _, batch_proposals_indices = torch.topk(batch_scores.squeeze(), k=self.num_proposal)
                batch_score_feats = batch_score_feats[batch_proposals_indices, :]
                batch_scores = batch_scores[batch_proposals_indices]
            if number_of_object_proposals_in_batch < self.num_proposal:
                proposal_mask[x, number_of_object_proposals_in_batch:self.num_proposal] = 0
                batch_score_feats = F.pad(batch_score_feats, [0, 0, 0, self.num_proposal - number_of_object_proposals_in_batch])
                batch_scores = F.pad(batch_scores, [0, 0, 0, self.num_proposal - number_of_object_proposals_in_batch])
            batch_score_feats = batch_score_feats * batch_scores
            batch[x] = batch_score_feats

        # print(f"batch.shape: {batch.shape}")
        batch = batch.cuda()
        data_dict['proposal_mask'] = proposal_mask
        data_dict['gt_proposals'] = gt_proposals
        data_dict['pg_loss'] = ret['pg_loss']
        data_dict = self.rfnet(data_dict['locs'], batch, data_dict)

        return data_dict
