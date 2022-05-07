import torch
from .point_head_template import PointHeadTemplate
from ...utils import box_utils



class PointHeadDouble_Aux(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.aux_cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.FP_CLS_FC,
            input_channels=64,
            output_channels=num_class
        )

        self.aux_part_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.FP_PART_FC,
            input_channels=64,
            output_channels=3
        )

        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        aux_coords = input_dict['aux_coords']
        gt_boxes = input_dict['gt_boxes']

        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        assert aux_coords.shape.__len__() in [2], 'aux_coords.shape=%s' % str(aux_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        targets_dict_aux = self.assign_stack_targets(
            points=aux_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=True
        )

        targets_dict['aux_cls_labels'] = targets_dict_aux['point_cls_labels']
        targets_dict['aux_part_labels'] = targets_dict_aux['point_part_labels']

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        aux_loss_part, tb_dict_2 = self.get_aux_part_layer_loss()
        aux_loss_cls, tb_dict_3 = self.get_aux_cls_layer_loss()

        point_loss = point_loss_cls + aux_loss_part + aux_loss_cls
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        tb_dict.update(tb_dict_3)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_features_FP: [V, C]
                point_coords_FP: [V, 4] (bs_idx, x, y, z)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.training:
            aux_features = batch_dict['aux_features']
            aux_cls_preds = self.aux_cls_layers(aux_features)  # (total_points, num_class)
            aux_part_preds = self.aux_part_reg_layers(aux_features)

        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {'point_cls_preds': point_cls_preds}
        if self.training:
            ret_dict['aux_cls_preds'] = aux_cls_preds
            ret_dict['aux_part_preds'] = aux_part_preds

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['aux_cls_labels'] = targets_dict['aux_cls_labels']
            ret_dict['aux_part_labels'] = targets_dict.get('aux_part_labels')
        
        self.forward_ret_dict = ret_dict

        return batch_dict
