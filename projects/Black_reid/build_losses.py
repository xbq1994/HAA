from fastreid.modeling import losses as Loss
from torch import nn
import torch
import pdb
def reid_losses(cfg, pred_class_logits, global_features, gt_classes, cross_scale, tri_scale, prefix='') -> dict:
    loss_dict = {}
    if pred_class_logits is not None:
        loss = getattr(Loss, cfg.MODEL.LOSSES.NAME[0])(cfg)(pred_class_logits, global_features, gt_classes)
        loss_dict.update(loss)
    if global_features is not None:
        loss = getattr(Loss, cfg.MODEL.LOSSES.NAME[1])(cfg)(pred_class_logits, global_features, gt_classes)
        loss_dict.update(loss)
    # rename
    named_loss_dict = {}
    for name in loss_dict.keys():
        if name == 'loss_cls':
            named_loss_dict[prefix + name] = loss_dict[name]*cross_scale
        if name == 'loss_triplet':
            named_loss_dict[prefix + name] = loss_dict[name]*tri_scale
    del loss_dict
    return named_loss_dict

def iou_losses(grid_list, head_x1, head_y1, head_x2, head_y2):
    iouloss_dict={}
    iou_loss = torch.nn.MSELoss(reduce=True, size_average=True)
    box_1_1 = grid_list[:,0,0,:].squeeze()
    box_1_2 = grid_list[:,-1,-1,:].squeeze()
    box_1 = torch.cat((box_1_1,box_1_2),dim=1)
    head = torch.cat(( head_x1.view(head_x1.shape[0],-1).long(), head_y1.view(head_y1.shape[0],-1).long(), head_x2.view(head_x2.shape[0],-1).long(), head_y2.view(head_y2.shape[0],-1).long()),dim=1)
    box_2 = 2*head-1
    box_2 = box_2.float()
    loss_iou = iou_loss(box_1,box_2)
    iouloss_dict['head_iouloss'] = loss_iou
    return iouloss_dict