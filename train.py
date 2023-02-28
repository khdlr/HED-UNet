"""
Usecase 3 Training Script

Usage:
    train.py [options]

Options:
    -h --help          Show this screen
    --summary          Only print model summary and return (Requires the torchsummary package)
    --resume=CKPT      Resume from checkpoint
    --config=CONFIG    Specify run config to use [default: config.yml]
"""
import sys, shutil, random, yaml
from warnings import warn
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from lib.data import get_loader
from lib.utils import mean, sample_map
from lib.plotting import log_image
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
import wandb
from einops import rearrange

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError as e:
    from torch.optim import Adam

from lib import get_loss, get_model, Metrics, flatui_cmap


@sample_map
def calculate_loss(prediction, target, kind):
  return loss_functions[kind](prediction, target)


def downscale(level, target, kind):
  target = rearrange(target, 'H W -> 1 1 H W')
  if kind in ['Fronts', 'Kochtitzky', 'Termpicks']:
    target = F.max_pool2d(target.float(), 1 << level)
    return rearrange(target, '1 1 sH sW -> sH sW')
  elif kind in ['Mask']:
    target = F.one_hot(target.long(), num_classes=2).float()
    target = rearrange(target, '1 1 H W C -> 1 C H W')
    target = F.avg_pool2d(target.float(), 1 << level)
    target = rearrange(target, '1 C H W -> C H W')
    return target
  elif kind in ['Zones']:
    target = F.one_hot(target.long(), num_classes=4).float()
    target = rearrange(target, '1 1 H W C -> 1 C H W')
    target = F.avg_pool2d(target, 1 << level)
    target = rearrange(target, '1 C H W -> C H W')
    return target
  else:
    raise ValueError(f'Unknown data kind {kind!r}')


def full_forward(model, data, metrics):
    data = tree_map(lambda x: x.to(dev), data)

    inputs, targets = [], []
    for sample in data:
      inp = {}
      outp = {}
      for k, v in sample.items():
        if k in loss_functions:
          outp[k] = v
        elif k in model.init:
          inp[k] = v
        else:
          raise ValueError(f'Not sure what to do with data of kind {k!r}')
      inputs.append(inp)
      targets.append(outp)

    y_hat, y_hat_levels = model(inputs)

    losses = defaultdict(list)
    deep_losses = [defaultdict(list) for _ in y_hat_levels]

    for i, sample_targets in enumerate(targets):
      for kind, target in sample_targets.items():
        pred = y_hat[kind][i]
        losses[kind].append(loss_functions[kind](pred, target))
        metrics.update_terms(pred, target, kind)

        # Deep Supervision Losses
        for level, deep_outs in enumerate(reversed(y_hat_levels)):
          if level == 0:
            small_target = target
          else:
            small_target = downscale(level, target, kind)

          small_pred = deep_outs[kind][i]
          deep_loss = loss_functions[kind](small_pred, small_target)
          deep_losses[level][kind].append(deep_loss)

    loss_terms = {
        k: mean(v) for k, v in losses.items()
    }
    full_loss = loss = mean(loss_terms.values())
    loss_terms['Loss'] = loss

    for level, dl in enumerate(reversed(deep_losses)):
      deep_loss = mean(mean(s) for s in dl.values())
      loss_terms[f'DeepLoss_{level}'] = deep_loss
      full_loss += deep_loss

    metrics.step(loss_terms)
    return dict(data=data, y_hat=y_hat, loss=full_loss)


def train(loader):
    global epoch
    # Training step

    epoch += 1
    model.train(True)
    prog = tqdm(loader, desc=f'Trn {epoch:02d}')
    for i, (data, metadata) in enumerate(prog): 
        for param in model.parameters():
            param.grad = None
        res = full_forward(model, data, metrics)
        res['loss'].backward()
        opt.step()

    metrics_vals = metrics.evaluate()
    wandb.log({f'trn/{k}': v for k, v in metrics_vals.items()}, step=epoch)

    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


@torch.no_grad()
def val(data_loader):
  # Validation step
  model.train(False)
  idx = 0

  current_image = None
  outputs = defaultdict(list)
  def log_image_wrapper(tile_id):
    if tile_id is None: return
    if not (epoch == 1 or epoch % 5 == 0):
      return
    tensors, meta = zip(*outputs[tile_id])
    log_image(tensors, meta, tile_id, step=epoch)
    del outputs[tile_id]

  for data, metadata in tqdm(data_loader, desc=f'Val {epoch:02d}'):
    res = full_forward(model, data, metrics)

    if epoch == 1 or epoch % 5 == 0:
      for i in range(len(data)):
        tile_id = metadata[i]['source_file'].stem
        if tile_id != current_image:
          log_image_wrapper(current_image)
        current_image = tile_id
        tensors = {
          **{f'Pred{k}': v[i] for k, v in res['y_hat'].items()},
          **data[i],
        }
        tensors = tree_map(lambda x: x.detach().cpu().numpy(), tensors)
        outputs[tile_id].append((tensors, metadata[i]))

  log_image_wrapper(current_image)

  metrics_vals = metrics.evaluate()
  wandb.log({f'val/{k}': v for k, v in metrics_vals.items()}, step=epoch)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    config = yaml.safe_load(open('config.yml'))

    modelclass = get_model(config['model'])
    model = modelclass(**config['model_args'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {dev} device')
    model = model.to(dev)

    epoch = 0
    metrics = Metrics()

    lr = config['learning_rate']
    opt = Adam(model.parameters(), lr)

    stack_height = 1 if 'stack_height' not in config['model_args'] else \
            config['model_args']['stack_height']

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False, parents=True)

    shutil.copy('config.yml', log_dir / 'config.yml')

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    trn_loader = get_loader(config, 'train')
    val_loader = get_loader(config, 'val')

    loss_functions = {}
    for kind, spec in config['model_args']['output_spec'].items():
      loss_functions[kind] = get_loss(spec['loss'])
      if type(loss_functions[kind]) is torch.nn.Module:
          loss_function = loss_functions[kind].to(dev)
 
    wandb.init(project=f'Multitask HED-UNet', config=config)
    for epoch in range(config['epochs']):
      wandb.log({'epoch': epoch}, step=epoch)
      train(trn_loader)
      val(val_loader)
