import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_type):
    if loss_type == 'AutoCE':
        return auto_weight_ce
    else:
        raise ValueError(f"No Loss of type {loss_type} known")

    return loss_class()


# @torch.jit.script
def auto_weight_ce(y_hat_logit, y):
    # Edge loss
    y_edge = y[:, 0].float()
    y_hat_edge = y_hat_logit[:, 0]
    with torch.no_grad():
        beta = y_edge.mean(dim=[1, 2], keepdims=True)
    logit_1 = F.logsigmoid(y_hat_edge)
    logit_0 = F.logsigmoid(-y_hat_edge)
    edge_loss = -(1 - beta) * logit_1 * y_edge \
                - beta * logit_0 * (1 - y_edge)
    edge_loss = torch.mean(edge_loss)

    y_seg = y[:, 1]
    y_hat_seg = y_hat_logit[:, 1:]

    seg_loss = nn.CrossEntropyLoss(ignore_index=-1)(y_hat_seg, y_seg)

    return edge_loss + seg_loss
