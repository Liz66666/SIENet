import torch
import torch.nn as nn
import torch.nn.functional as F

import os 
from ..model_utils import pytorch_utils as pt_utils

class PCN_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = pt_utils.Conv1d(3, 128) 
        self.mlp2 = pt_utils.Conv1d(128, 256, activation=None) 
        self.mlp3 = pt_utils.Conv1d(512, 512) 
        self.mlp4 = pt_utils.Conv1d(512, 1024, activation=None) 

    def forward(self, input_pts, pts_in_roi):
        inputs_BCL = input_pts.permute(0, 2, 1) # [1, 3, M]
        features = self.mlp1(inputs_BCL)
        features = self.mlp2(features)    # [1, 256, M]
               
        # max pooling	 
        new_features = []
        for f in features.split(list(pts_in_roi), dim=2):
            value, _ = torch.max(f, dim=2)
            new_features.append(value)

        pooled_features_of_sample = torch.cat(new_features, dim=0).unsqueeze(2)    # [B, 256, 1] global feature g
        
        # expand
        expand_features_cache = torch.split(pooled_features_of_sample, 1, dim=0)
        expand_features_out = [f.expand(1, 256, pts_in_roi[i]) for i,f in enumerate(expand_features_cache)]

        features_global = torch.cat(expand_features_out, dim=2) # [1, 256, M] global feature of each point
        features = torch.cat([features, features_global], dim=1) # [1, 512, M] 
        
        features = self.mlp3(features)
        features = self.mlp4(features) # [1, 1024, M]
        
        # max pooling
        new_features = []
        for f in features.split(list(pts_in_roi), dim=2):
            value, _ = torch.max(f, dim=2)
            new_features.append(value)
        
        global_feature_v = torch.cat(new_features, dim=0)    # global feature v

        return global_feature_v

class PCN_dncoder(nn.Module):
    def __init__(self, num_coarse, num_fine, grid_size, grid_scale):
        super().__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.grid_size = grid_size
        self.grid_scale = grid_scale

        self.fc1 = pt_utils.FC(1024, 1024)
        self.fc2 = pt_utils.FC(1024, 1024)
        self.fc3 = pt_utils.FC(1024, self.num_coarse * 3, activation=None)

        # we remove the refinement branch!
        # self.mlp1 = pt_utils.Conv1d(2+3+1024, 512, bn=True)
        # self.mlp2 = pt_utils.Conv1d(512, 512, bn=True)
        # self.mlp3 = pt_utils.Conv1d(512, 3, activation=None)
        
    def forward(self, features):
        coarse = self.fc1(features)
        coarse = self.fc2(coarse)
        coarse = self.fc3(coarse) # [B, num_coarse * 3] 
        coarse = coarse.view(-1, self.num_coarse, 3) # [B, num_coarse, 3] 

        return coarse

        # now folding
        # x = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size).cuda(non_blocking=True)
        # y = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size).cuda(non_blocking=True)
        # grid = torch.meshgrid(x, y)
        # grid = torch.stack(grid, dim=2).view(-1, 2).unsqueeze(0)  # (1, 16, 2)

        # grid_feat = grid.repeat(batch_size, self.num_coarse, 1)   # (B, 16*num_coarse, 2)
        # point_feat = coarse.unsqueeze(2).repeat(1, 1, self.grid_size ** 2, 1) # (B, num_coarse, 16, 3) 
        # point_feat = point_feat.view(-1, self.num_fine, 3)    # (B, num_fine, 3)

        # global_feat = features.unsqueeze(1).repeat(1, self.num_fine, 1) # (B, num_fine, 1024) 
        # feat = torch.cat([grid_feat, point_feat, global_feat], dim=2) # (B, 16384, 2+3+1024)

        # center = coarse.unsqueeze(2).repeat(1, 1, self.grid_size ** 2, 1) # (B, num_coarse, 16, 3)
        # center = center.view(-1, self.num_fine, 3) # (B, num_fine, 3)

        # feat = feat.permute(0, 2, 1)
        # fine = self.mlp1(feat) 
        # fine = self.mlp2(fine)
        # fine = self.mlp3(fine)  # (B, 3, num_fine)
        # fine = fine.permute(0, 2, 1)  # (B, num_fine, 3)

        # fine = fine + center 
        # return coarse, fine
        
        


class PointCompletionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # reproduced according to the PCN paper, we remove the refine branch to save GPU consuming
        # PCN: Point Completion Network, 3DV 2018, Link: https://www.cs.cmu.edu/~wyuan1/pcn/#paper
        self.num_coarse = 1024 
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse 
        self.encoder = PCN_encoder()
        self.decoder = PCN_dncoder(self.num_coarse, self.num_fine, self.grid_size, self.grid_scale)

        # freeze all parameters
        for p in self.parameters():
            p.requires_grad = False   
    
    def forward(self, input_pts, pts_in_roi):
        with torch.no_grad():
            global_feature_v = self.encoder(input_pts, pts_in_roi) # [B, 1024]
            coarse_out = self.decoder(global_feature_v) # [B, num_coarse, 3] 

        return coarse_out