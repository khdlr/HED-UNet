import torch
from torch.utils.data import Dataset
from pathlib import Path


class PTDataset(Dataset):
    """
    Random access Dataset for datasets of pytorch tensors stored like this:
        data/images/1.pt
        data/images/2.pt
        ...
        data/masks/1.pt
        data/masks/2.pt
        ...
    """
    def __init__(self, root, parts, transform=None, suffix='.pt'):
        self.root = Path(root)
        self.parts = parts

        first = self.root / parts[0]
        filenames = list(sorted([x.name for x in first.glob('*' + suffix)]))
        self.index = [[self.root / p / x for p in parts] for x in filenames]
        self.transform = transform

    def __getitem__(self, idx):
        files = self.index[idx]
        data = [torch.load(f) for f in files]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.index)


class Augment(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = 8 * len(self.dataset)

    def __getitem__(self, idx):
        idx, carry = divmod(idx, 8)
        carry, flipx = divmod(carry, 2)
        transpose, flipy = divmod(carry, 2)

        diry = 2 * flipy - 1
        dirx = 2 * flipx - 1
        base = self.dataset[idx]
        augmented = []
        for field in base:
            field = field.numpy()
            field = field[:, ::diry, ::dirx]
            if transpose == 1:
                field = field.transpose(0, 2, 1)
            augmented.append(torch.from_numpy(field.copy()))

        # Channel jitter
        sar = augmented[0]
        # jitter = 0.5 + torch.rand(2)
        # augmented[0] = sar * jitter.reshape(-1, 1, 1)

        return tuple(augmented)

    def __len__(self):
        return len(self.dataset) * 8
