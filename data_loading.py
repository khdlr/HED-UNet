import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from deep_learning.utils.data import PTDataset, Augment
from pathlib import Path

def transform_fn(sample, eps=1e-7):
    data, mask = sample
    HH, HV = data[0], data[1]
    data = torch.log(torch.clamp(torch.stack([HH, HV], dim=0), min=1e-4))

    mask = mask.to(torch.float)
    return data, mask


def _get_dataset(dataset, names=['scene', 'mask'], augment=False):
    ds_path = 'data/scenes/' + dataset
    dataset = PTDataset(ds_path, names, transform=transform_fn)
    if augment:
        dataset = Augment(dataset)
    return dataset


def get_loader(folders, batch_size, num_workers=0, augment=False, shuffle=False, names=['scene', 'mask']):
    folders = [_get_dataset(ds, names=names, augment=augment) for ds in folders]
    concatenated = ConcatDataset(folders)
    return DataLoader(concatenated, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True)


def get_batch(data_names):
    base_dir = Path('data/scenes')
    data = []
    for sample in data_names:
        scene, tile = sample.split('/')
        tensor_file = base_dir / scene / 'scene' / f'{tile}.pt'
        tensor = torch.load(tensor_file)
        mask_file = base_dir / scene / 'mask' / f'{tile}.pt'
        mask = torch.load(mask_file)

        data.append(transform_fn((tensor, mask)))

    out = []
    for tensors in zip(*data):
        out.append(torch.stack(tensors, dim=0))

    return out
