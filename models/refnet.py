import torch
import torch.nn as nn
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
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps", use_lang_classifier=True):
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
        batch_size =ret["batch_offsets"].shape[0] -1
        print(f"scores {scores.shape}")
        print(f"score_feats {score_feats.shape}")
        print(f"proposals_idx {proposals_idx.shape}")
        print(f"proposals_offset {proposals_offset.shape}")
        print(f"batch_offsets {batch_offsets.shape}")
        print(f"batch_size {batch_size}")

        scores = scores.cuda()
        score_feats = score_feats.cuda()
        proposals_idx = proposals_idx.cuda()
        proposals_offset = proposals_offset.cuda()
        batch_offsets = batch_offsets.cuda()

        batch = torch.zeros(batch_size, 128, 16 )
        for i in range(batch_size):
            prop_offs_batch = proposals_offset.clone()
            prop_offs_batch[ prop_offs_batch  >= batch_offsets[i+1]] = 0
            prop_offs_batch[ prop_offs_batch < batch_offsets[i]] = 0
            min = prop_offs_batch.detach().nonzero().squeeze().min()
            max = prop_offs_batch.detach().nonzero().squeeze().max()
            print(f"min: {min}")
            print(f"max: {max}")
            n_prop = max-min
            batch_scores = scores[min:max]
            batch_score_feats = score_feats[min:max, :]
            _, batch_proposals_indices = torch.topk(batch_scores.squeeze(), k=(128 if n_prop >= 128 else n_prop))
            batch_score_feats = batch_score_feats[batch_proposals_indices, :] * batch_scores[batch_proposals_indices]
            if n_prop< 128:
                padding = torch.zeros(128-n_prop, 16).cuda()
                batch_score_feats = torch.cat((batch_score_feats, padding))
            batch[i] = batch_score_feats

        print(f"batch.shape: {batch.shape}")
        batch = batch.cuda()
        data_dict = self.rfnet(data_dict['locs'], batch, data_dict)

        return data_dict
