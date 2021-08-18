import torch
import numpy as np
import cv2


def get_parameter_number(net):
    parameter_list = [p.numel() for p in net.parameters()]
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def weighted_f1_loss(gt, pre, beta2=0.3):
    y_true, y_pre = gt.clone(), torch.sigmoid(pre.clone())
    if len(y_true.shape) == 3:
        y_true = y_true.view([y_true.shape[0], 1, *y_true.shape[1:]])
    assert y_pre.shape == y_true.shape
    prec, recall = _eval_pr(y_pre, y_true, 255)
    f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
    f_score[f_score != f_score] = 0  # for Nan
    return f_score.max().cuda()


def _eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall


def get_edge_by_Canny(img, kernel_size, low=0, high=60):
    edge = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), low, high)
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    dilated_edge = cv2.dilate(edge, kernel, iterations=1)
    return dilated_edge


def get_edge_by_Laplace(img, kernel_size):
    [gy, gx] = np.gradient(img)
    edge = gy * gy + gx * gx
    edge[edge != 0.] = 1.
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    edge = cv2.dilate(edge, kernel, iterations=1)
    return edge


def get_current_edge(sa_pre):
    # b 1 h w
    sa_pre[sa_pre > 0.5] = 1
    sa_pre[sa_pre != 1] = 0
    sa_pre = sa_pre * 255.
    sa_pre = sa_pre.astype(np.uint8)
    b, _, h, w = sa_pre.shape
    ret = np.zeros((b, 1, h, w), dtype=np.float32)
    for i in range(b):
        ret[i, 0] = get_edge_by_Laplace(sa_pre[i, 0], 1)
    ret[ret > 0] = 1
    return ret
