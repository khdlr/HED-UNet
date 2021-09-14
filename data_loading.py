import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from deep_learning.utils.data import InriaDataset, Augment
from pathlib import Path

def transform_fn(sample, eps=1e-7):
    data, mask = sample
    data = (data.to(torch.float) / 127.) - 1.
    mask = mask.to(torch.float)
    return data, mask


def get_dataset(dataset, names=['images', 'gt'], augment=False):
    ds_path = 'data/AerialImageDataset/' + dataset
    dataset = InriaDataset(ds_path, names, transform=transform_fn)
    if augment:
        dataset = Augment(dataset)
    return dataset
