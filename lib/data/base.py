from typing import Iterator
import xarray
import torch.utils.data
from collections import defaultdict
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from abc import abstractmethod
from torch.utils._pytree import tree_map

from .utils import md5


class GlacierFrontDataset(torch.utils.data.Dataset):
  def __init__(self, config, mode):
    super().__init__()
    self.config = config
    self.root = Path(config['data_root'])
    self.tilesize = config['tilesize']
    self.mode = mode
    self.__post_init__()
    if not self.cache_path().exists():
      self.build_cache()
    self.data = xarray.open_dataset(self.cache_path(), cache=False)
    self.variables = [var for var in self.data.variables if not var.endswith('_present')]

  def __post_init__(self):
    ...

  def cache_path(self):
    return self.root / 'cache' / f'{self.cache_tag()}_{self.tilesize}x{self.tilesize}_{self.mode}.nc'

  def build_cache(self):
    path = self.cache_path()
    if not path.parent.exists():
      path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
      tiles = defaultdict(list)
      present = defaultdict(list)

      count = 0
      for data_tiles in self.generate_tiles():
        for kind, data in data_tiles.items():
          while len(tiles[kind]) < count:
            tiles[kind].append(np.zeros_like(data))
          tiles[kind].append(data)

          while len(present[kind]) < count:
            present[kind].append(False)
          present[kind].append(True)
        count += 1

      for kind in tiles:
        while len(tiles[kind]) < count:
          tiles[kind].append(np.zeros_like(tiles[kind][0]))
        while len(present[kind]) < count:
          present[kind].append(False)

      arrays = {}
      for kind in tiles:
        print(f'Stacking {kind}')
        stacked = np.stack(tiles[kind])
        if stacked.ndim == 4:
          dims = ['sample', f'{kind}_channel', 'x', 'y']
        elif stacked.ndim == 3:
          dims = ['sample', 'x', 'y']
        else:
          raise ValueError()
        print(kind, stacked.shape)
        arrays[kind] = xarray.DataArray(stacked, dims=dims)
        arrays[f'{kind}_present'] = xarray.DataArray(np.asarray(present[kind]), dims=['sample'])

      data = xarray.Dataset(arrays)
      data.to_netcdf(self.cache_path(), engine='h5netcdf')

  def generate_tiles(self) -> Iterator[dict[str, np.ndarray]]:
    size = self.tilesize
    for data in self.generate_images():
      print('Got:', tree_map(lambda x: x.shape, data))
      *_, H, W = list(data.values())[0].shape

      ysteps = 1 + H // size
      xsteps = 1 + W // size
      for y in np.linspace(0, H-size, ysteps):
        y = round(y)
        for x in np.linspace(0, W-size, xsteps):
          x = round(x)
          yield {k: v[..., y:y+size, x:x+size] for k, v in data.items()}

  @abstractmethod
  def cache_tag(self) -> str:
    ...

  @abstractmethod
  def generate_images(self) -> Iterator[dict[str, np.ndarray]]:
    ...

  def __getitem__(self, idx):
    out = {}
    for key in self.variables:
      if self.data[f'{key}_present'][idx]:
        out[key] = self.data[key][idx]
    return out

  def __len__(self):
    return len(self.data.sample)
