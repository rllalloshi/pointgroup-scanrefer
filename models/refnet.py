import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.ref_module import RefModule


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
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim-1)

        # Hough voting
        # self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation, detection and language reference
        self.rfnet = RefModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, use_lang_classifier)

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

        batch_scene_objects = data_dict["gt_scene_objects"]

        batch_size = batch_scene_objects.shape[0]

        batch_features = np.zeros((batch_size, 128, 128))
        for i in range(batch_size):
            xyz, features = self.backbone_net._break_up_pc(batch_scene_objects[i])
            data_dict['xyz_gt'] = xyz
            data_dict['features_gt'] = features
            data_dict = self.backbone_net(data_dict)
            features = data_dict["fp2_features"].detach().cpu().numpy()
            features = np.resize(features, (features.shape[0], features.shape[1]))
            batch_features[i] = features
        data_dict = self.rfnet(batch_features, data_dict)
                
        # --------- HOUGH VOTING ---------

        # data_dict["seed_inds"] = data_dict["fp2_inds"]
        # data_dict["seed_xyz"] = xyz
        # data_dict["seed_features"] = features
        #
        # xyz, features = self.vgen(xyz, features)
        # features_norm = torch.norm(features, p=2, dim=1)
        # features = features.div(features_norm.unsqueeze(1))
        # data_dict["vote_xyz"] = xyz
        # data_dict["vote_features"] = features



        return data_dict
