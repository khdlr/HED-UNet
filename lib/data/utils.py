import hashlib
import torch
from torch.utils.data import Dataset

def md5(obj):
    obj = str(obj).encode("utf8")
    return hashlib.md5(obj).hexdigest()[:16]


def list_collate(batch):
  """ Essentially a transpose operation """
  return tuple(zip(*batch))


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
