import torch
import numpy as np
from collections import namedtuple
from .detectors import build_detector


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            if key in ['frame_id']:
                continue
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

# added for PCDet_v1.0
def example_convert_to_torch(example, dtype=torch.float32):
    device = torch.cuda.current_device()
    example_torch = {}
    float_names = [
        'voxels', 'anchors', 'box_reg_targets', 'reg_weights', 'part_labels',
        'gt_boxes', 'voxel_centers', 'reg_src_targets', 'points',
    ]

    for k, v in example.items():
        if k in float_names:
            try:
                example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(dtype)
            except RuntimeError:
                example_torch[k] = torch.zeros((v.shape[0], 1, 7), dtype=torch.float32, device=device).to(dtype)
        elif k in ['coordinates', 'box_cls_labels', 'num_points', 'seg_labels']:
            example_torch[k] = torch.tensor(v, dtype=torch.int32, device=device)
        else:
            example_torch[k] = v
    return example_torch