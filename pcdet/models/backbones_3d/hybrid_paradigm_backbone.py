import torch
import torch.nn as nn
import spconv
from functools import partial
from ...utils import common_utils
from ...ops.pointnet2_SASSD import pointnet2_utils as pt2_utils

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class Auxiliary_branch(nn.Module):
    def __init__(self, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size
    
    def tensor2points(self, tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
        indices = tensor.indices.float() # [V, 4]
        offset = torch.Tensor(offset).to(indices.device)
        voxel_size = torch.Tensor(voxel_size).to(indices.device)
        indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size

        return tensor.features, indices

    def nearest_neighbor_interpolate(self, unknown, known, known_feats):
        """
        :param unknown: (n, 4) tensor of the bxyz positions of the unknown features
        :param known: (m, 4) tensor of the bxyz positions of the known features
        :param known_feats: (m, C) tensor of features to be propigated
        :return:
            unknown_feats: (n, C) tensor of the features of the unknown features
        """
        dist, idx = pt2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pt2_utils.three_interpolate(known_feats, idx, weight)

        return interpolated_feats
    
    def forward(self, x, points_mean):
        x_fp_ft, x_fp_xyz = self.tensor2points(x, self.voxel_size)
        x_fp = self.nearest_neighbor_interpolate(points_mean, x_fp_xyz, x_fp_ft)

        return x_fp

class HybridParadigmBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01) # set BN1d

        self.sparse_shape = grid_size[::-1] + [1, 0, 0] # [41, 1600, 1408]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        ) # [B, Cin, 41, 1600, 1408] -> [B, 16, 41, 1600, 1408]
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        ) # [B, 16, 41, 1600, 1408] -> [B, 16, 41, 1600, 1408]

        self.conv2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        ) # [B, 16, 41, 1600, 1408] -> [B, 32, 21, 800, 704]

        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        ) # [B, 32, 21, 800, 704] -> [B, 64, 11, 400, 352]

        self.conv4 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        ) # [B, 64, 11, 800, 704] -> [B, 64, 5, 200, 176]

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        ) # [B, 64, 5, 200, 176] -> [B, 128, 2, 200, 176]
        
        # for auxiliary branch
        self.fp1 = Auxiliary_branch(voxel_size=(.1, .1, .2))
        self.fp2 = Auxiliary_branch(voxel_size=(.2, .2, .4))
        self.fp3 = Auxiliary_branch(voxel_size=(.4, .4, .8))

        self.point_fc = nn.Linear(64+64+32, 64, bias=False)

        self.num_point_features = 128
        self.num_aux_features = 64
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                points: [43901, 5]  (NP, 1 + 3 + C) [bs_idx, x, y, z, ...]                                                                                                     | 1/1856 [00:02<1:31:03,  2.95s/it, total_it=1]
                frame_id: B
                gt_boxes torch.Size: [B, max_boxes, 8]
                use_lead_xyz torch.Size([B])
                voxels: [V, T, C]
                voxel_num_points: [V]
                image_shape: [B, 2]
                batch_size: int
                voxel_features: [V, 4] (V is total number of voxels in this batch) (x, y, z, r)
                voxel_coords: [V, 4] (batch_idx, z_idx, y_idx, x_idx)
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = voxel_coords[:, 0].int()
        points_mean[:, 1:] = voxel_features[:, :3]
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        if self.training:
            x_fp1 = self.fp1(x_conv2, points_mean)

        x_conv3 = self.conv3(x_conv2)
        if self.training:
            x_fp2 = self.fp2(x_conv3, points_mean)

        x_conv4 = self.conv4(x_conv3)
        if self.training:
            x_fp3 = self.fp3(x_conv4, points_mean)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        if not self.training:
            return batch_dict
        
        # for auxiliary branch 
        seg_features = self.point_fc(torch.cat([x_fp1, x_fp2, x_fp3], dim=-1))
                
        batch_dict['aux_features'] = seg_features
        point_coords = common_utils.get_voxel_centers(
            x_conv1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['aux_coords'] = torch.cat((x_conv1.indices[:, 0:1].float(), point_coords), dim=1)

        return batch_dict
