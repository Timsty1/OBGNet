import torch
import torch.nn.functional as F


def bce2d_new(input, target, reduction='mean'):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
