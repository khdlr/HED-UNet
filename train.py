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
from datetime import datetime
from pathlib import Path
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from einops import reduce

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError as e:
    from torch.optim import Adam

from deep_learning import get_loss, get_model, Metrics, flatui_cmap


def showexample(idx, img, target, prediction):
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=m, wspace=m)
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw=gridspec_kw)
    heatmap_seg  = dict(cmap='tab20', vmin=0.5, vmax=19.5)
    heatmap_edge = dict(cmap=flatui_cmap('Clouds', 'Midnight Blue'), vmin=0, vmax=1)
    # Clear all axes
    for axis in ax.flat:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    rgb = img.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    rgb = rgb * std + mean
    ax[0, 0].imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    ax[0, 1].imshow(target[1].cpu(), **heatmap_seg)
    ax[1, 1].imshow(target[0].cpu(), **heatmap_edge)

    seg_pred = torch.argmax(prediction[1:], dim=0)
    edge_pred = torch.sigmoid(prediction[0])
    ax[0, 2].imshow(seg_pred.cpu(), **heatmap_seg)
    ax[1, 2].imshow(edge_pred.cpu(), **heatmap_edge)

    filename = log_dir / 'figures' / f'{idx:03d}_{epoch}.jpg'
    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def get_pyramid(mask):
    with torch.no_grad():
        is_edge = (mask == 255).long()
        mask[mask > 200] = -1
        label = torch.stack([is_edge, mask], dim=1)
        pyramid = [label]
        # Build mip-maps
        for _ in range(stack_height):
            # Pretend we have a batch
            big_label   = pyramid[-1]
            is_edge     = (reduce(big_label[:,0], 'b (h h2) (w w2) -> b h w', 'min', h2=2, w2=2) == 0)
            small_label = big_label[:, 1, ::2, ::2]
            pyramid.append(torch.stack([is_edge, small_label], dim=1))

    return pyramid


def full_forward(model, img, target, metrics):
    img = img.to(dev)
    target = target.to(dev)
    y_hat, y_hat_levels = model(img)
    target = get_pyramid(target)
    loss_levels = []
    
    for y_hat_el, y in zip(y_hat_levels, target):
        loss_levels.append(loss_function(y_hat_el, y))
    # Overall Loss
    loss_final = loss_function(y_hat, target[0])
    # Pyramid Losses (Deep Supervision)
    loss_deep_super = torch.sum(torch.stack(loss_levels))
    loss = loss_final + loss_deep_super

    target = target[0]
    seg_pred = torch.argmax(y_hat[:, 1:], dim=1)
    seg_acc = (seg_pred == target[:, 1]).float().mean()

    edge_pred = (y_hat[:, 0] > 0).float()
    edge_acc = (edge_pred == target[:, 0]).float().mean()

    metrics.step(Loss=loss, SegAcc=seg_acc, EdgeAcc=edge_acc)

    return dict(
        img=img,
        target=target,
        y_hat=y_hat,
        loss=loss,
        loss_final=loss_final,
        loss_deep_super=loss_deep_super
    )


def train(dataset):
    global epoch
    # Training step

    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=True, num_workers=config['data_threads'],
        pin_memory=True
    )

    epoch += 1
    model.train(True)
    for img, target in tqdm(data_loader):
        for param in model.parameters():
            param.grad = None
        res = full_forward(model, img, target, metrics)
        res['loss'].backward()
        opt.step()

    metrics_vals = metrics.evaluate()
    logstr = f'Epoch {epoch:02d} - Train: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)

    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


@torch.no_grad()
def val(dataset):
    # Validation step
    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=False, num_workers=config['data_threads'],
        pin_memory=True
    )

    model.train(False)
    idx = 0
    for img, target in data_loader:
        B = img.shape[0]
        res = full_forward(model, img, target, metrics)

        for i in range(B):
            if idx+i in config['visualization_tiles']:
                showexample(idx+i, img[i], res['target'][i], res['y_hat'][i])
        idx += B

    metrics_vals = metrics.evaluate()
    logstr = f'Epoch {epoch:02d} - Val: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    print(logstr)
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)


class DataTransform:
    def __init__(self, augment=False):
        self.augment = augment
        self.jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

    def __call__(self, img, target):
        img = TF.resize(img, 256)
        target = TF.resize(target, 256, interpolation=Image.NEAREST)

        if self.augment:
            i = random.randrange(5)
            img = TF.five_crop(img, 224)[i]
            target = TF.five_crop(target, 224)[i]

            if random.random() < .5:
                img = TF.hflip(img)
                target = TF.hflip(target)

        else:
            img = TF.center_crop(img, 224)
            target = TF.center_crop(target, 224)

        img = TF.to_tensor(img)
        target = torch.from_numpy(np.array(target)).long()

        if self.augment:
            img = self.jitter(img)

        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, target


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

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

    epoch = 0
    metrics = Metrics()

    lr = config['learning_rate']
    opt = Adam(model.parameters(), lr)

    if cli_args['--summary']:
        from torchsummary import summary
        summary(model, [(3, 256, 256)])
        sys.exit(0)

    stack_height = 1 if 'stack_height' not in config['model_args'] else \
            config['model_args']['stack_height']

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False, parents=True)

    shutil.copy(config_file, log_dir / 'config.yml')

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    trn_dataset = datasets.VOCSegmentation('data/', image_set='train',
            download=True, transforms=DataTransform(augment=True))
    trn_dataset = torch.utils.data.ConcatDataset([trn_dataset] * 10)
    val_dataset = datasets.VOCSegmentation('data/', image_set='val',
            download=True, transforms=DataTransform(augment=False))

    loss_function = get_loss(config['loss_function'])
    if type(loss_function) is torch.nn.Module:
        loss_function = loss_function.to(dev)

    for _ in range(config['epochs']):
        train(trn_dataset)
        val(val_dataset)
