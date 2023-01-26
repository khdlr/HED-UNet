import re
import torch.nn
import torch.nn.functional as F

def get_loss(loss_args):
    loss_type = loss_args['type']
    functional_style = re.search(r'(\w+)\((\w+)\)', loss_type)
    args = dict()
    if functional_style:
        func, arg = functional_style.groups()
        new_args = dict(loss_args)
        if func == 'Summed':
            new_args['type'] = arg
            return sum_loss(get_loss(new_args))
    if loss_type == 'BCE':
        loss_class = torch.nn.BCEWithLogitsLoss
        if 'pos_weight' in loss_args:
            args['pos_weight'] = loss_args['pos_weight'] * torch.ones([])
    elif loss_type == 'FocalLoss':
        return focal_loss_with_logits
    elif loss_type == 'AutoBCE':
        return auto_weight_bce
    else:
        raise ValueError(f"No Loss of type {loss_type} known")

    return loss_class(**args)


def focal_loss_with_logits(y_hat_log, y, gamma=2):
    log0 = F.logsigmoid(-y_hat_log)
    log1 = F.logsigmoid(y_hat_log)

    gamma0 = torch.pow(torch.abs(1 - y - torch.exp(log0)), gamma)
    gamma1 = torch.pow(torch.abs(y - torch.exp(log1)), gamma)

    return torch.mean(-(1 - y) * gamma0 * log0 - y * gamma1 * log1)


def auto_weight_bce(y_hat_log, y):
    with torch.no_grad():
        beta = y.mean(dim=[2, 3], keepdims=True)
    logit_1 = F.logsigmoid(y_hat_log)
    logit_0 = F.logsigmoid(-y_hat_log)
    loss = -(1 - beta) * logit_1 * y \
           - beta * logit_0 * (1 - y)
    return loss.mean()

def sum_loss(loss_fn):
    def loss(prediction, target):
        if type(prediction) is list:
            losses = torch.stack([loss_fn(p, t) for p, t in zip(prediction, target)])
            return torch.sum(losses)
        else:
            return loss_fn(prediction, target)
    return loss
