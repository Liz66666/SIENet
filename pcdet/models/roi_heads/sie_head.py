import torch.nn as nn
import torch
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from .PCN import PointCompletionNetwork as PCN
from ...ops.pointnet2_lib.pointnet2_fast import pointnet2_utils as pt2
from ..model_utils import pytorch_utils as pt_utils
import torch.nn.functional as F


class attentive_fusion(nn.Module):
    # reproduced perspective-channel fusion according to 3D iou-net
    # 3D iou-net: Iou guided 3D object detector for point clouds (arXiv:2004.04962)
    def __init__(self, pre_channels=144):
        super().__init__()
        self.pre_channels = pre_channels
        self.pre_points = 6*6*6

        self.point_wise_attention = nn.Sequential(
            nn.Linear(in_features=self.pre_points, out_features=self.pre_points), 
            nn.Linear(in_features=self.pre_points, out_features=self.pre_points),
            nn.ReLU(), 
        )
        self.channel_wise_attention = nn.Sequential(
            nn.Linear(in_features=self.pre_channels, out_features=self.pre_channels), 
            nn.Linear(in_features=self.pre_channels, out_features=self.pre_channels),
            nn.ReLU(), 
        )
   
    def forward(self, features):
        # features [B, P, C]
        point_features = F.max_pool2d(features, kernel_size=[1, features.size(2)]).squeeze(-1) # [B, P]
        channel_features = F.max_pool2d(features, kernel_size=[features.size(1), 1]).squeeze(1) # [B, C]

        point_attention = self.point_wise_attention(point_features).unsqueeze(-1)
        channel_attention = self.channel_wise_attention(channel_features).unsqueeze(1)
        
        attention = point_attention * channel_attention
        attention = F.sigmoid(attention) # [B, P, C]
        
        features = attention * features

        return features

class SIEHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        
        # build spatial shape prediction network
        self.pcn = PCN()
        
        # for structure information extraction
        self.pcn_group1 = pt2.QueryAndGroup(self.model_cfg.PCN.RADIUS[0], self.model_cfg.PCN.NSAMPLE[0], use_xyz=True)
        self.pcn_mlp1 = nn.Sequential(
            pt_utils.Conv2d(self.model_cfg.PCN.MLP[0][0], self.model_cfg.PCN.MLP[0][1], bn=True),
            pt_utils.Conv2d(self.model_cfg.PCN.MLP[0][1], self.model_cfg.PCN.MLP[0][2], bn=True),
        )
        self.pcn_group2 = pt2.QueryAndGroup(self.model_cfg.PCN.RADIUS[1], self.model_cfg.PCN.NSAMPLE[1], use_xyz=True)
        self.pcn_mlp2 = nn.Sequential(
            pt_utils.Conv2d(self.model_cfg.PCN.MLP[1][0], self.model_cfg.PCN.MLP[1][1], bn=True),
            pt_utils.Conv2d(self.model_cfg.PCN.MLP[1][1], self.model_cfg.PCN.MLP[1][2], bn=True),
        )
        self.pcn_down = nn.Sequential(
            pt_utils.FC(self.model_cfg.PCN.DOWN[0], self.model_cfg.PCN.DOWN[1], bn=True),
            pt_utils.FC(self.model_cfg.PCN.DOWN[1], self.model_cfg.PCN.DOWN[2], bn=True),
            )
        
        # for feature fusion net
        self.attentive_fusion = attentive_fusion()

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )
        
        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(
            out_size=self.model_cfg.ROI_AWARE_POOL.POOL_SIZE,
            max_pts_each_voxel=self.model_cfg.ROI_AWARE_POOL.MAX_POINTS_PER_VOXEL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * (c_out + 16)

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

        self.load_pretrained_sspn()

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Rx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1]) # [B*R, 8]
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B*R, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points
    
    def prepare_data(self, batch_dict):
        batch_size = batch_dict['batch_size']
        all_voxel_points = batch_dict['voxel_features'][...,:3] # [V, 3] (x, y, z)
        all_voxel_batch_id = batch_dict['voxel_coords'][...,0]  # [B] (b_idx)
        rois = batch_dict['rois'] # [B, R, 7]

        pooled_points_list = []

        for bs_idx in range(batch_size):
            bs_mask = (all_voxel_batch_id == bs_idx)
            cur_voxel_points = all_voxel_points[bs_mask] # [V_0, 3]
            cur_roi = rois[bs_idx][:, 0:7].contiguous()  # [R, 7]

            pooled_points = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_voxel_points, cur_voxel_points, pool_method='avg'
            )  # (R, out_x, out_y, out_z, 3)
            
            pooled_points_list.append(pooled_points)

        pooled_points = torch.cat(pooled_points_list, dim=0)  # (B*R, out_x, out_y, out_z, 3)

        # generate mask 
        sparse_idx = pooled_points.sum(dim=-1).nonzero()  # [NE, 4] ==> [bs_idx, x_idx, y_idx, z_idx]
        pcn_mask = torch.zeros((pooled_points.shape[0], pooled_points.shape[1], pooled_points.shape[2], pooled_points.shape[3], 1), 
                        device=pooled_points.device, dtype=torch.float)
        pcn_mask[sparse_idx[:, 0], sparse_idx[:, 1], sparse_idx[:, 2], sparse_idx[:, 3]] = 1.0 
        
        # Coordinate System Transformation
        pooled_points = pooled_points.view(pooled_points.shape[0], -1, pooled_points.shape[-1]) # (B*R, out_x*out_y*out_z, 3)
        batch_rois = rois.view(-1, rois.shape[-1]) # [B*R, 8]
        roi_center = batch_rois[:, 0:3].unsqueeze(1) # [B*R, 1, 3]
        pooled_points -= roi_center
        pooled_points = common_utils.rotate_points_along_z(pooled_points, -batch_rois[:, 6])
        pooled_points = pooled_points / batch_rois[:, 3].unsqueeze(1).unsqueeze(1)
        
        # note that the pre-trained completion network uses openpcdet's (x, z, -y)
        pooled_points = pooled_points[:, :, [0, 2, 1]]
        pooled_points[:, :, 2] = -1.0*pooled_points[:, :, 2]
        pooled_points = pooled_points.view(
            pooled_points.shape[0], self.model_cfg.ROI_AWARE_POOL.POOL_SIZE, 
            self.model_cfg.ROI_AWARE_POOL.POOL_SIZE, self.model_cfg.ROI_AWARE_POOL.POOL_SIZE, 3)

        # add mask
        pooled_points = pooled_points * pcn_mask
        pooled_points = pooled_points.view(pooled_points.shape[0], -1, 3)

        return pooled_points
    
    def load_pretrained_sspn(self):
        from collections import OrderedDict 
        import os
        from pathlib import Path
        new_sicn_dict = OrderedDict()

        filename = Path(self.model_cfg.PCN.PRETRAINED_MODEL)
        print (filename)
        if os.path.isfile(filename):
            print ("==> Loading Point Completion model from checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
            sicn_net_dict = self.pcn.state_dict()

            pretrained_dict = checkpoint['model_state']
            for k,v in pretrained_dict.items(): 
                new_sicn_dict[k] = v

            # find dicts 
            useful_sicn_dict = {k: v for k, v in new_sicn_dict.items() if k in sicn_net_dict}
            # then update them
            sicn_net_dict.update(useful_sicn_dict)
            
            self.pcn.load_state_dict(sicn_net_dict)
        
            for k,v in useful_sicn_dict.items():
                print ('sspn: {} is loaded'.format(k))

        else:
            raise FileNotFoundError
    
    def pcn_forward(self, pooled_inputs):
        with torch.no_grad():
            pooled_inputs_cache = pooled_inputs.clone().detach()

            # [NE, 2] (0 for B*R-dim idx, 1 for 2744-dim idx)
            sparse_idx = pooled_inputs.sum(dim=-1).nonzero()

            # counts number of points in each RoI
            # if a RoI includes no point, manually give it a fake point
            pts_num = []
            for bi in range(pooled_inputs_cache.shape[0]):
                cur_batch_ne_num = (torch.sum(sparse_idx[:, 0].int() == bi)).item()
                if cur_batch_ne_num == 0:
                    pooled_inputs_cache[bi, 0, :] = 1e-5
                    cur_batch_ne_num = 1
                pts_num.append(cur_batch_ne_num)

            # [NE1, 2] (0 for B*R-dim idx, 1 for 2744-dim idx)
            new_sparse_idx = pooled_inputs_cache.sum(dim=-1).nonzero()
            # [1, NE1, 3] (NE1 non-empty points)
            non_empty_pts = pooled_inputs_cache[new_sparse_idx[:, 0], new_sparse_idx[:, 1]].unsqueeze(0)
            # [B*R, PO, 3] [B*R, 1024]          
            coarse_out = self.pcn.forward(non_empty_pts, pts_num)

            return coarse_out

    def extract_pcn_out(self, pcn_in, pcn_out):
        pcn_in.requires_grad = self.training
        pcn_out.requires_grad = self.training

        # select keypoint index [B*R, KP]
        key_point_index = pt2.furthest_point_sample(pcn_out, self.model_cfg.PCN.FPS_NUM)
        # select keypoint [B*R, KP, 3]
        key_point = pt2.gather_operation(pcn_out.permute(0, 2, 1).contiguous(), key_point_index).permute(0, 2, 1)
        
        ball_feature1 = self.pcn_group1(pcn_out.contiguous(), key_point.contiguous())  
        ball_feature1 = self.pcn_mlp1(ball_feature1)
        ball_feature1 = F.max_pool2d(ball_feature1, kernel_size=[1, ball_feature1.size(3)]).squeeze(-1) # [B*R, C, KP]
        ball_feature2 = self.pcn_group2(pcn_out.contiguous(), key_point.contiguous())  
        ball_feature2 = self.pcn_mlp2(ball_feature2)
        ball_feature2 = F.max_pool2d(ball_feature2, kernel_size=[1, ball_feature2.size(3)]).squeeze(-1) # [B*R, C, KP]
        
        ball_feature = torch.cat([ball_feature1, ball_feature2], dim=1)
        ball_feature = ball_feature.view(ball_feature.shape[0], -1) # [B*R, C*KP]
        
        # [B*R, 16]
        ball_feature = self.pcn_down(ball_feature)
    
        return ball_feature       

    def forward(self, batch_dict):
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # spatial shape prediction network
        pcn_in = self.prepare_data(batch_dict) # (BxR, 14*14*14, 3)
        pcn_out = self.pcn_forward(pcn_in) # (BxR, 1024, 3)
        
        # feature extraction
        pcn_feature = self.extract_pcn_out(pcn_in, pcn_out) # (BxR, 16)
        
        # RoI grid pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxR, 6x6x6, C)

        # fusion
        pcn_feature = pcn_feature.unsqueeze(1).repeat(1, 6*6*6, 1)
        pooled_features = torch.cat([pooled_features, pcn_feature], dim=-1)
        pooled_features = self.attentive_fusion(pooled_features)
        
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C+16, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict