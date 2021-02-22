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
import sys
import distutils.util
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from deep_learning.models import get_model
from deep_learning.loss_functions import get_loss
from deep_learning.metrics import Metrics, Accuracy, Precision, Recall, F1, IoUWater, IoULand, mIoU
from data_loading import get_loader, get_batch

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import re

from docopt import docopt
import yaml


def showexample(idx, filename):
    ## First plot
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=m, wspace=m)
    N = 1 + int(np.ceil(len(vis_predictions) / 3))
    fig, ax = plt.subplots(3, N, figsize=(3*N, 9), gridspec_kw=gridspec_kw)
    ax = ax.T.reshape(-1)
    heatmap_rb = dict(cmap='coolwarm', vmin=0, vmax=1)
    heatmap_bw = dict(cmap='binary_r', vmin=0, vmax=1)

    batch_img, *batch_targets = vis_batch
    batch_img = batch_img.to(torch.float)

    # Clear all axes
    for axis in ax:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    rgb = batch_img[idx, [0, 0, 0]].cpu().numpy()
    ax[0].imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    ax[0].axis('off')

    ax[1].imshow(batch_targets[0][idx, 0].cpu(), **heatmap_rb)
    ax[1].axis('off')

    for i, pred in enumerate(vis_predictions):
        if config['model'].startswith('HED') and type(pred) is tuple:
            pred, pred_lvl = pred
        ax[i+3].imshow(pred[idx, 0].cpu(), **heatmap_rb)
        ax[i+3].axis('off')

    filename.parent.mkdir(exist_ok=True)
    overview_file = filename.with_name(f'_Overview_{filename.name}')
    plt.savefig(overview_file, bbox_inches='tight')
    plt.close()

    if not config['model'].startswith('HED'):
        return
    if config['model'] == 'HEDUNet' and not model.deep_supervision:
        return

    ## Second plot
    N = len(vis_predictions[-1][-1]) + 1
    fig, ax = plt.subplots(4, N, figsize=(3*N, 12), gridspec_kw=gridspec_kw)
    ax = ax.T

    # Clear all axes
    for axis in ax.flat:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    pred = vis_predictions[-1][0]
    rgb = batch_img[idx, [0, 0, 0]].cpu().numpy()
    ax[0, 0].imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    if 'seg' in model.tasks:
        ax[0, 2].imshow(pred[idx, model.tasks.index('seg')].cpu(), **heatmap_rb)
    if 'edge' in model.tasks:
        ax[0, 3].imshow(pred[idx, model.tasks.index('edge')].cpu(), **heatmap_bw)

    for i, (pred, target) in enumerate(zip(vis_predictions[-1][-1], batch_targets)):
        if 'seg' in model.tasks:
            ax[i+1, 0].imshow(target[idx, model.tasks.index('seg')].cpu(), **heatmap_rb)
            ax[i+1, 1].imshow(pred[idx, model.tasks.index('seg')].cpu(), **heatmap_rb)
        if 'edge' in model.tasks:
            ax[i+1, 2].imshow(pred[idx, model.tasks.index('edge')].cpu(), **heatmap_bw)
            ax[i+1, 3].imshow(target[idx, model.tasks.index('edge')].cpu(), **heatmap_bw)

    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename.with_suffix(f'.{epoch}.jpg'), bbox_inches='tight')
    plt.close()


def scoped_get(key, *scopestack):
    for scope in scopestack:
        value = scope.get(key)
        if value is not None:
            return value
    raise ValueError(f'Could not find "{key}" in any scope.')


def get_dataloader(name):
    if name in dataset_cache:
        return dataset_cache[name]
    if name in config['datasets']:
        ds_config = config['datasets'][name]
        ds_config['num_workers'] = config['data_threads']
        if 'batch_size' not in ds_config:
            ds_config['batch_size'] = config['batch_size']
        dataset_cache[name] = get_loader(**ds_config)
        return dataset_cache[name]
    else:
        func, arg = re.search(r'(\w+)\((\w+)\)', name).groups()
        return COMMANDS[func](arg)
    return dataset_cache[name]


def get_pyramid(mask):
    with torch.no_grad():
        masks = [mask]
        ## Build mip-maps
        for _ in range(stack_height):
            # Pretend we have a batch
            big_mask = masks[-1]
            small_mask = F.avg_pool2d(big_mask, 2)
            masks.append(small_mask)

        targets = []
        for mask in masks:
            sobel = torch.any(SOBEL(mask) != 0, dim=1, keepdims=True).float()
            if config['model'] == 'HED':
                targets.append(sobel)
            else:
                targets.append(torch.cat([mask, sobel], dim=1))

    return targets


def train(dataset):
    global epoch
    # Training step
    data_loader = get_dataloader(dataset)

    epoch += 1
    model.train(True)
    for iteration, (img, target) in enumerate(tqdm(data_loader)):
        img = img.to(dev)
        target = target.to(dev)

        y_hat = model(img)

        opt.zero_grad()
        if 'edge' in model.tasks:
            target = get_pyramid(target)
            loss_levels = []
            if type(y_hat) is tuple:
                y_hat, y_hat_levels = y_hat
                for y_hat_el, y in zip(y_hat_levels, target):
                    loss_levels.append(loss_function(y_hat_el, y))
            # Overall Loss
            loss = loss_function(y_hat, target[0])
            # Pyramid Losses (Deep Supervision)
            if loss_levels:
                full_loss = loss + torch.sum(torch.stack(loss_levels))
            else:
                full_loss = loss
            full_loss.backward()
        else:
            loss = loss_function(y_hat, target[:, [0]])
            loss.backward()
        opt.step()

        if config['model'] == 'HRNet_OCR':
            y_hat = y_hat[-1]
            y_hat = y_hat[:, [1]] - y_hat[:, [0]]
            y_hat = F.interpolate(input=y_hat, size=img.shape[2:],
                    mode='bilinear', align_corners=True)

        with torch.no_grad():
            lvl = {}
            if 'edge' in model.tasks:
                lvl = {f'{i}': loss_levels[i].detach() for i in range(len(loss_levels))}
                y_hat = y_hat[:,[0]]
                target = target[0][:,[0]]
            metrics.step(y_hat, target, Loss=loss.detach(), **lvl)

    metrics_vals = metrics.evaluate()
    logstr = f'Epoch {epoch:02d} - Train: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)

    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


def val(dataset):
    # Validation step
    data_loader = get_dataloader(dataset)

    model.train(False)
    with torch.no_grad():
        for iteration, (img, target) in enumerate(data_loader):
            img = img.to(dev)

            if config['model'] == 'HED':
                target = get_pyramid(target.to(dev))[0]
            else:
                target = target[:, [0]].to(dev)
            y_hat = model(img)
            # if 'edge' in model.tasks and config['model'] not in ('GSCNN', 'DexiNed'):
            if 'edge' in model.tasks and type(y_hat) is tuple:
                # TODO: Edge Detection Metrics
                y_hat, _ = y_hat
            if config['model'] != 'HRNet_OCR':
                y_hat = y_hat[:, [0]]
            else:
                y_hat = y_hat[-1]
                y_hat = y_hat[:, [1]] - y_hat[:, [0]]
                y_hat = F.interpolate(input=y_hat, size=img.shape[2:],
                        mode='bilinear', align_corners=True)
            loss = loss_function(y_hat, target)
            metrics.step(y_hat, target, Loss=loss.detach())

    metrics_vals = metrics.evaluate()
    logstr = f'Epoch {epoch:02d} - Val: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    print(logstr)
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)


def log_images():
    model.train(False)
    with torch.no_grad():
        res = model(vis_imgs)
        if config['model'].startswith('HED') and type(res) is tuple:
            res = (res[0].cpu(), [t.cpu() for t in res[1]])
        elif config['model'] == 'HRNet_OCR':
            res = res[-1]
            res = res[:, [1]] - res[:, [0]]
            res = F.interpolate(input=res, size=vis_imgs.shape[2:],
                    mode='bilinear', align_corners=True)
        else:
            res = res.cpu()
        vis_predictions.append(res)
    for i, tile in enumerate(config['visualization_tiles']):
        filename = log_dir / f"{tile.replace('/', '_')}.jpg"
        filename.parent.mkdir(exist_ok=True)
        showexample(i, filename)


def hard_samples(dataset):
    data = get_dataloader(dataset).dataset

    straight_loader = DataLoader(data, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    print("Mining hard examples...")
    with torch.no_grad():
        accs = []
        for idx, img, label in tqdm(straight_loader):
            img = img.to(dev)
            label = label.to(dev, non_blocking=True)
            pred, *_ = model(img)
            acc = ((pred[:,0] > 0.5) == label[:,0]).float().mean(axis=2).mean(axis=1)
            accs.append(acc)
        accs = torch.cat(accs).cpu()
        worst_fraction = 0.5
        vals, indices = accs.topk(int(worst_fraction * len(accs)), largest=False)
        print(f"Selecting hard examples with accuracy < {vals.max()}")

    subset = Subset(data, indices)
    return DataLoader(subset, batch_size=config['batch_size'], num_workers=config['data_threads'],
            shuffle=True, pin_memory=True)


def prioritized(dataset, s=100):
    data = get_dataloader(dataset).dataset
    argsort = torch.argsort(torch.argsort(last_score)).float()
    # weights = 1 / (1 + argsort.float())
    scale = - np.log(s) / len(data)
    weights = torch.exp(scale * argsort)
    sampler = WeightedRandomSampler(weights, replacement=True, num_samples=len(data) // 2)
    return DataLoader(data, batch_size=config['batch_size'], num_workers=config['data_threads'],
            sampler=sampler, pin_memory=True)


COMMANDS = dict(
    train_on=train,
    validate_on=val,
    log_images=log_images,
    hard_samples=hard_samples,
    prioritized=prioritized
)

SOBEL = nn.Conv2d(1, 2, 1, padding=1, padding_mode='replicate', bias=False)
SOBEL.weight.requires_grad = False
SOBEL.weight.set_(torch.Tensor([[
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]],
   [[-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]]).reshape(2, 1, 3, 3))


if __name__ == "__main__":
    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config_file = Path(cli_args['--config'])
    config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)

    modelclass = get_model(config['model'])
    model = modelclass(**config['model_args'])

    if cli_args['--resume']:
        config['resume'] = cli_args['--resume']

    if 'resume' in config and config['resume']:
        checkpoint = Path(config['resume'])
        if not checkpoint.exists():
            raise ValueError(f"There is no Checkpoint at {config['resume']} to resume from!")
        if checkpoint.is_dir():
            # Load last checkpoint in run dir
            ckpt_nums = [int(ckpt.stem) for ckpt in checkpoint.glob('checkpoints/*.pt')]
            last_ckpt = max(ckpt_nums)
            config['resume'] = checkpoint / 'checkpoints' / f'{last_ckpt:02d}.pt'
        print(f"Resuming training from checkpoint {config['resume']}")
        model.load_state_dict(torch.load(config['resume']))

    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    SOBEL = SOBEL.to(dev)

    epoch = 0
    train_metrics = {}
    val_metrics = {}
    vis_predictions = []

    metrics = Metrics(Accuracy, Precision, Recall, F1, IoUWater, IoULand, mIoU)

    lr = config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if cli_args['--summary']:
        from torchsummary import summary
        summary(model, [(7, 256, 256)])
        sys.exit(0)

    dataset_cache = {}
    last_score = torch.zeros(len(get_dataloader('train').dataset))

    stack_height = 1 if 'stack_height' not in config['model_args'] else \
            config['model_args']['stack_height']
    vis_imgs, vis_mask = get_batch(config['visualization_tiles'])
    vis_targets = [t.cpu() for t in get_pyramid(vis_mask.to(dev))]
    vis_batch = [vis_imgs, *vis_targets]
    vis_imgs = vis_imgs.to(dev)

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False)

    shutil.copy(config_file, log_dir / 'config.yml')

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    for phase in config['schedule']:
        print(f'Starting phase "{phase["phase"]}"')
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(f'Phase {phase["phase"]}', file=f)
        for _ in range(phase['epochs']):
            # Epoch setup
            loss_function = get_loss(scoped_get('loss_function', phase, config))
            try:
                loss_function = loss_function.to(dev)
            except:
                pass

            datasets_config = scoped_get('datasets', phase, config)

            for step in phase['steps']:
                if type(step) is dict:
                    assert len(step) == 1
                    (command, arg), = step.items()
                    COMMANDS[command](arg)
                else:
                    command = step
                    COMMANDS[command]()
