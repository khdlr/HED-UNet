from torch.utils.data import ConcatDataset
from .utils import Augment
from .sar import SARDataset


def get_dataset(config, mode, augment=False):
  sar_part = SARDataset(config, mode)

  dataset = ConcatDataset([sar_part])
  if augment:
      dataset = Augment(dataset)
  return dataset

