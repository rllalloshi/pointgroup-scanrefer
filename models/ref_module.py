""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import os
import sys
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder

class RefModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, use_lang_classifier=True, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.seed_feat_dim = seed_feat_dim

        # --------- FEATURE FUSION ---------
        self.gru = nn.GRU(
            input_size=300,
            hidden_size=256,
            batch_first=True
        )
        self.lang_sqz = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.feat_fuse = nn.Sequential(
            nn.Conv1d(in_channels=128 + 128, out_channels=128, kernel_size=1),
            nn.ReLU()
        )

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(128, 18),
                nn.Dropout()
            )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128,1,1),
            nn.Dropout()
        )

    def forward(self, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        # --------- FEATURE FUSION ---------
        lang_feat = data_dict["lang_feat"]
        lang_feat = pack_padded_sequence(lang_feat, data_dict["lang_len"], batch_first=True, enforce_sorted=False)
    
        # encode description
        _, lang_feat = self.gru(lang_feat)
        data_dict["lang_emb"] = lang_feat
        lang_feat = self.lang_sqz(lang_feat.squeeze(0)).unsqueeze(2).repeat(1, 1, self.num_proposal)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(lang_feat[:, :, 0])
        
        # fuse
        features = self.feat_fuse(torch.cat([torch.from_numpy(features).float().cuda(), lang_feat], dim=1))

        objectness_scores = data_dict['gt_scene_objects_mask']
        data_dict['objectness_scores'] = objectness_scores
        data_dict['cluster_ref'] = self.conv4(features).squeeze(1)
        
        return data_dict

