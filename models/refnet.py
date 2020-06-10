import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
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
        self.object_classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_class),
        ).to(torch.device("cuda"))

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
        num_objects = batch_scene_objects.shape[1]
        num_points = batch_scene_objects.shape[2]
        num_features = batch_scene_objects.shape[3]

        batch = batch_scene_objects.view(batch_size*num_objects,num_points, num_features)

        xyz, features = self.backbone_net._break_up_pc(batch)

        data_dict['xyz_gt'] = xyz
        data_dict['features_gt'] = features

        data_dict = self.backbone_net(data_dict)

        batch_features = data_dict["fp2_features"].view(batch_size, num_objects, -1)
        data_dict['seed_xyz'] = data_dict['fp2_xyz'].squeeze(1).view(batch_size, num_objects, -1).detach().cpu().numpy()
        data_dict['seed_inds'] = data_dict["fp2_inds"].view(batch_size, num_objects, -1).detach().cpu().numpy()

        output_classifier = self.object_classifier.forward(batch_features.view(batch_features.shape[0] * batch_features.shape[1], batch_features.shape[2]))
        data_dict['object_classifier'] = output_classifier.view(batch_features.shape[0],  batch_features.shape[1], self.num_class)
        data_dict = self.rfnet(batch_features, data_dict)

        return data_dict
