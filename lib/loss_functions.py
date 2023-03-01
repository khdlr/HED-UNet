import torch
import torch.nn
import torch.nn.functional as F

def no_batch(loss_fn):
  def inner(x, y):
    return loss_fn(x.unsqueeze(0), y.unsqueeze(0))
  return inner

def get_loss(loss_type):
  if loss_type == 'BCE':
    loss_fn = torch.nn.BCEWithLogitsLoss()
  elif loss_type == 'FocalLoss':
    loss_fn = focal_loss_with_logits
  elif loss_type == 'AutoBCE':
    loss_fn = auto_weight_bce
  elif loss_type == 'NoSigmoidAutoBCE':
    loss_fn = no_sigmoid_auto_weight_bce
  elif loss_type == 'CrossEntropy':
    loss_fn = cross_entropy
  else:
    raise ValueError(f"No Loss of type {loss_type} known")

  return no_batch(loss_fn)


def focal_loss_with_logits(y_hat_log, y, gamma=2):
    log0 = F.logsigmoid(-y_hat_log)
    log1 = F.logsigmoid(y_hat_log)

    gamma0 = torch.pow(torch.abs(1 - y - torch.exp(log0)), gamma)
    gamma1 = torch.pow(torch.abs(y - torch.exp(log1)), gamma)

    return torch.mean(-(1 - y) * gamma0 * log0 - y * gamma1 * log1)


def cross_entropy(y_hat_log, y):
  if y.ndim < y_hat_log.ndim:
    y = y.long()  # Target contains indices
  return F.cross_entropy(y_hat_log, y)


def no_sigmoid_auto_weight_bce(y_hat_log, y):
  y = y.float()
  if y.ndim < y_hat_log.ndim:
    y = y.unsqueeze(1)

  with torch.no_grad():
      beta = y.mean(dim=[2, 3], keepdims=True)
  logit_1 = y_hat_log
  logit_0 = torch.log(1 - torch.exp(y_hat_log))
  loss = -(1 - beta) * logit_1 * y \
         - beta * logit_0 * (1 - y)
  return loss.mean()


def auto_weight_bce(y_hat_log, y):
  y = y.float()
  if y.ndim < y_hat_log.ndim:
    y = y.unsqueeze(1)

  with torch.no_grad():
      beta = y.mean(dim=[2, 3], keepdims=True)
  logit_1 = F.logsigmoid(y_hat_log)
  logit_0 = F.logsigmoid(-y_hat_log)
  loss = -(1 - beta) * logit_1 * y \
         - beta * logit_0 * (1 - y)
  return loss.mean()

