import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.pointnet2.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.input_feature_dim = input_feature_dim

        # --------- 4 SET ABSTRACTION LAYERS ---------
        self.sa1 = PointnetSAModuleMSG(
            npoint=128,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 64],
            mlps=[[input_feature_dim, 32, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]],
            use_xyz=True,
        )

        input_channels = 64 + 128 + 128
        self.sa2 = PointnetSAModuleMSG(
            npoint=64,
            radii=[0.2, 0.4, 0.8],
            nsamples=[8, 16, 32],
            mlps=[
                [input_channels, 64, 64, 128],
                [input_channels, 128, 128, 256],
                [input_channels, 128, 128, 256],
            ],
            use_xyz=True
        )
        self.sa3 = PointnetSAModule(
            mlp=[128 + 256 + 256, 256, 512, 128],
            use_xyz=True,
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, data_dict):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """

        xyz, features = data_dict["xyz_gt"], data_dict['features_gt']

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features = self.sa1(xyz, features)
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features

        xyz, features = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features

        xyz, features = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        #print(features.shape)
        data_dict['fp2_features'] = features
        return data_dict

if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
