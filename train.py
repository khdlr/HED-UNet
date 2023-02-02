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
from tqdm import tqdm
from lib.data import get_loader
from lib.utils import mean, sample_map

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
import wandb

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError as e:
    from torch.optim import Adam

from lib import get_loss, get_model, Metrics, flatui_cmap


@sample_map
def calculate_loss(prediction, target, kind):
  return loss_functions[kind](prediction, target)


def full_forward(model, data, metrics):
    data = tree_map(lambda x: x.to(dev), data)

    inputs, targets, output_kinds = [], [], []
    for sample in data:
      inp = {}
      outp = {}
      outp_kinds = []
      for k, v in sample.items():
        if k in loss_functions:
          outp[k] = v
          outp_kinds.append(k)
        elif k in model.init:
          inp[k] = v
        else:
          raise ValueError(f'Not sure what to do with data of kind {k!r}')
      inputs.append(inp)
      targets.append(outp)
      output_kinds.append(outp_kinds)

    y_hat, y_hat_levels = model(inputs, output_kinds)
    losses = calculate_loss(y_hat, targets)
    warn('No Deep Supervision yet')
    
    loss = mean(mean(sample_losses.values()) for sample_losses in losses)
    metrics.step(y_hat, targets, Loss=loss)

    return dict(data=data, loss=loss)


def train(loader):
    global epoch
    # Training step

    epoch += 1
    model.train(True)
    prog = tqdm(loader)
    for i, (data, metadata) in enumerate(prog): 
        for param in model.parameters():
            param.grad = None
        res = full_forward(model, data, metrics)
        res['loss'].backward()
        opt.step()

        if (i+1) % 100 == 0:
            prog.set_postfix(metrics.peek())

    metrics_vals = metrics.evaluate()
    wandb.log({f'trn/{k}': v for k, v in metrics_vals.items()}, step=epoch)

    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


@torch.no_grad()
def val(data_loader):
    # Validation step

    model.train(False)
    idx = 0
    for data, metadata in data_loader:
        res = full_forward(model, data, metrics)
        # TODO: Image logging

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
