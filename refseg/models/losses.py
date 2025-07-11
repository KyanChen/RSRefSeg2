from torch import nn
from mmseg.registry import MODELS
import torch
import torch.nn.functional as F

@MODELS.register_module()
class MILCrossEntropy(nn.Module):
    def __init__(self):
        super(MILCrossEntropy, self).__init__()

    def forward(self, pred_logits, target, dim=-1, weighted_unk=False, weights=None, avg_positives=False):
        # if weighted_unk:
        #     pred_logits[target == 2] /= weighted_unk

        # pred_logits B H W
        # target B H W
        pred_logits = pred_logits.flatten(1)
        target = target.flatten(1)

        target[target == 2] = 0
        probs = F.softmax(pred_logits, dim=-1)
        # only consider the valid targets
        valid_mask = torch.any(target > 0, dim=-1)
        probs = probs[valid_mask]
        target = target[valid_mask]
        if len(target) == 0:
            return torch.tensor(0., device=pred_logits.device)
        if avg_positives:  # average the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim) / (torch.sum(target, dim=dim) + 1e-6))
        else:  # sum the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim))
        if weights is not None:
            return (loss * weights).mean()
        return loss.mean()