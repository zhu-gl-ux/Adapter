import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    def __init__(self, ignore_index=255, ignore_bg=True, pos_weight=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.reduction = reduction

        if ignore_bg is True:
            self.ignore_indexes = [0, self.ignore_index]
        else:
            self.ignore_indexes = [self.ignore_index]

    def forward(self, logit, label, logit_old=None):
        # logit:     [N, C_tot, H, W]
        # logit_old: [N, C_prev, H, W]
        # label:     [N, H, W] or [N, C, H, W]
        C = logit.shape[1]
        if logit_old is None:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            elif len(label.shape) == 4:
                target = label
            else:
                raise NotImplementedError
            
            logit = logit.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.permute(0, 2, 3, 1).reshape(-1, C)

            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
        else:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                target[:, 1:logit_old.shape[1]] = logit_old.sigmoid()[:, 1:]
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            else:
                raise NotImplementedError
            
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
            del target

            return loss


class WBCELoss(nn.Module):
    def __init__(self, ignore_index=255, pos_weight=None, reduction='none', n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:t-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes  # |Ct|, 19-1: 1 | 15-5: 5 | 15-1: 1
        
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)
        
    def forward(self, logit, label):
        # logit:     [N, |Ct|, H, W]
        # label:     [N, H, W]
        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()
        for cls_idx in torch.clamp(label, min=0).unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )

        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        elif self.reduction == 'mean':
            return loss
        else:
            raise NotImplementedError


class PKDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, features, features_old, pseudo_label_region):

        pseudo_label_region_5 = F.interpolate(
            pseudo_label_region, size=features[-1].shape[2:], mode="bilinear", align_corners=False)
        loss_5 = self.criterion(features[-1], features_old[-1])
        loss_5 = (loss_5 * pseudo_label_region_5).sum() / (
                    (pseudo_label_region_5.sum() * features[-1].shape[1]) + 0.0001)
        # assert np.isnan(loss_5) is True,"pkd loss is nan, please ensure the value is avliabel"
        return loss_5
def calculate_certainty(seg_probs):
    """Calculate the uncertainty of segmentation probability

    Args:
        seg_probs (torch.Tensor): B x C x H x W
            probability map of segmentation

    Returns:
        torch.Tensor: B x 1 x H x W
            uncertainty of input probability
    """
    top2_scores = torch.topk(seg_probs, k=2, dim=1)[0]
    res = (top2_scores[:, 0].detach() - top2_scores[:, 1]).unsqueeze(1)
    return res

class ContLoss(nn.Module):
    def __init__(self, ignore_index=255, n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:t-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes

        self.criteria = nn.MSELoss(reduction='mean')

    def forward(self, features, logit, label,n_new_classes,prev_prototypes=None):

        target = torch.zeros_like(logit[:,-n_new_classes:], device=logit.device).float()
        pred = logit.argmax(dim=1) + 1  # pred: [N. H, W]
        idx = (logit > 0.5).float()  # logit: [N, C, H, W]
        idx = idx.sum(dim=1)  # logit: [N, H, W]
        pred[idx == 0] = 0  # set background (non-target class)
        target_neg = torch.zeros_like(logit[:,-n_new_classes:], device=logit.device).float()
        for cls_idx in label.unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()
            target_neg[:, int(cls_idx) - self.n_old_classes] = ((label != int(cls_idx)*(pred==int(cls_idx)))).float()

        small_target = F.interpolate(target, size=features.shape[2:], mode='bilinear', align_corners=False)
        # small_target_neg = F.interpolate(target_neg, size=features.shape[2:], mode='bilinear', align_corners=False)
        new_feature = F.normalize(features, p=2, dim=1).unsqueeze(1) * small_target.unsqueeze(2)
        # new_feature_neg = F.normalize(features, p=2, dim=1).unsqueeze(1) * small_target_neg.unsqueeze(2)

        new_center = F.normalize(new_feature.sum(dim=[0, 3, 4]), p=2, dim=1)
        # new_neg_center = F.normalize(new_feature_neg.sum(dim=[0, 3, 4]), p=2, dim=1)

        dist_pp = torch.norm(new_center.unsqueeze(0) - prev_prototypes.unsqueeze(1), p=2, dim=2)
        # dist_neg = torch.norm(new_neg_center.detach().unsqueeze(0) - new_center.unsqueeze(1), p=2, dim=2)
        # if dist_neg.min(0).values==0:
        #     l_nn = 0
        # else:
        #     l_nn = (1 / (dist_neg.min(0).values + 10)).mean()
        l_neg = (1 / dist_pp.min(0).values).mean()


        return l_neg
        # return l_neg+l_nn