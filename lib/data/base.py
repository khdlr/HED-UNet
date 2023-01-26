from typing import Iterator
import xarray
import torch.utils.data
from collections import defaultdict
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from abc import abstractmethod

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
      for data_tiles in self.generate_tiles():
        for kind in data_tiles:
          tiles[kind].append(data_tiles[kind])

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

      data = xarray.Dataset(arrays)
      data.to_netcdf(self.cache_path(), engine='h5netcdf')

  def generate_tiles(self) -> Iterator[dict[str, np.ndarray]]:
    size = self.tilesize
    for data in tqdm(self.generate_images()):
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
    for key in self.data.variables:
      out[key] = self.data[key][idx]

  def __len__(self):
    raise NotImplementedError()


if __name__ == "__main__":
    config = {
        "dataset": "TUD-MS",
        "data_root": "../aicore/uc1_new/",
        "vertices": 64,
        "tile_size": 256,
    }
    ds = GlacierFrontDataset("test_esa", config, subtiles=False)
    print(len(ds))
    for x in ds[0]:
        print(x.shape, x.dtype)
