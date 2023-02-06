import torch
import xarray
from math import ceil
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
from .utils import md5, list_collate
from warnings import warn
import numpy as np

class NCDataset(Dataset):
  def __init__(self, netcdf_path, tilesize, sampling_mode):
    self.netcdf_path = netcdf_path
    self.tilesize = tilesize
    self.data = xarray.open_dataset(netcdf_path, cache=False, decode_coords='all')

    self.sampling_mode = sampling_mode

    self.H, self.W = len(self.data.y), len(self.data.x)
    self.H_tile = ceil(self.H / self.tilesize)
    self.W_tile = ceil(self.W / self.tilesize)

  def __getitem__(self, idx):
    if self.sampling_mode == 'deterministic':
      y_tile, x_tile = divmod(idx, self.W_tile)
      y0 = round((self.H - self.tilesize) * y_tile / self.H_tile)
      x0 = round((self.W - self.tilesize) * x_tile / self.W_tile)
    elif self.sampling_mode == 'random':
      y0 = int(torch.randint(0, max(1, self.H - self.tilesize), ()))
      x0 = int(torch.randint(0, max(1, self.W - self.tilesize), ()))
    else:
      raise ValueError(f'Unsupported tiling mode: {self.sampling_mode!r}')
    y1 = y0 + self.tilesize
    x1 = x0 + self.tilesize


    tile = {}
    for k in self.data:
      patch = self.data[k][..., y0:y1, x0:x1]
      patch = patch.fillna(0).values
      patch, pad_params = self.ensure_tilesize(patch)
      patch = torch.from_numpy(patch)
      tile[k] = patch

    metadata = {
      'source_file': self.netcdf_path,
      'y0': y0, 'x0': x0,
      'y1': y1, 'x1': x1,
      **pad_params,
    }

    return tile, metadata

  def ensure_tilesize(self, img):
    *_, H, W = img.shape

    y_padding_needed = max(self.tilesize - H, 0)
    x_padding_needed = max(self.tilesize - W, 0)

    if y_padding_needed == 0 and x_padding_needed == 0:
      return img, dict(py0=0, py1=1, px0=0, px1=1)

    if self.sampling_mode == 'deterministic':
      y0 = y_padding_needed // 2
      x0 = x_padding_needed // 2
    elif self.sampling_mode == 'random':
      y0 = int(torch.randint(0, y_padding_needed + 1, ()))
      x0 = int(torch.randint(0, x_padding_needed + 1, ()))
    else:
      raise ValueError(f'Unsupported tiling mode: {self.sampling_mode!r}')

    y1 = y_padding_needed - y0
    x1 = x_padding_needed - x0
    pad_seq = [(y0, y1), (x0, x1)]
    while len(pad_seq) < img.ndim:
      pad_seq = [(0, 0)] + pad_seq
    padded = np.pad(img, pad_seq)
    return padded, dict(py0=y0, py1=y1, px0=x0, px1=x1)

  def __len__(self):
    return self.H_tile * self.W_tile

  def __del__(self):
    try:
      self.data = self.data.close()
    except:
      print('Caught something in 1')
      pass
    try:
      del self.data
    except:
      print('Caught something in 2')
      pass


def val_filter(filepath):
  return md5(filepath.stem)[-1] in ['a', '3']


def get_loader(config, mode):
  root = Path(config['data_root'])
  sampling_mode = 'random' if mode == 'train' else 'deterministic'

  if mode == 'test':
    root = root / 'cubes' / 'test'
  elif mode in ['train', 'val', 'tinytrain']:
    root = root / 'cubes' / 'train'
  else:
    raise ValueError(f'Dataset mode {mode!r} not supported!')
  ncs = list(sorted(root.glob('*.nc')))
  if mode == 'train':
    ncs = [nc for nc in ncs if not val_filter(nc)]
  elif mode == 'tinytrain':
    ncs = [nc for nc in ncs if not val_filter(nc)][::400]
  elif mode == 'val':
    ncs = [nc for nc in ncs if val_filter(nc)]

  datasets = [NCDataset(nc, config['tile_size'], sampling_mode)
                for nc in tqdm(ncs, desc=f'Opening {mode}')]
  full_data = ConcatDataset(datasets)

  warn('No Data Augmentation yet!')

  shuffle = mode in ('train', 'tinytrain')

  return DataLoader(full_data,
                    shuffle=shuffle,
                    batch_size=config['batch_size'],
                    num_workers=config['data_threads'],
                    collate_fn=list_collate)
  

if __name__ == '__main__':
  import yaml
  from PIL import Image
  from torch.utils._pytree import tree_map
  import numpy as np
  from einops import rearrange

  torch.random.manual_seed(0)
  config = yaml.safe_load(open('config.yml'))
  config['data_threads'] = 4

  for mode in ['tinytrain', 'train', 'val', 'test']:
    loader = get_loader(config, mode)
    print(mode, 'samples:', len(loader.dataset), 'scenes:', len(loader.dataset.datasets))
    for batch, meta in loader:
      for i, sample in enumerate(batch):
        print(i, sample.keys())
        for kind in sample: 
          data = sample[kind].numpy()
          print(f'  {kind} {data.min():.02f} -- {data.max():.02f}')
          img = np.clip(255 * data, 0, 255).astype(np.uint8)
          if img.ndim == 3:
            if img.shape[0] > 3:
              img = img[[3,2,1]]
            img = rearrange(img, 'C H W -> H W C')
            if img.shape[-1] == 1:
              img = img[..., 0]

          Image.fromarray(img).save(f'viz/{i:02d}_{kind}.png')

      break
    break
  # for batch in loader:
  #   print(tree_map(lambda x: x.shape, batch))
  #   break

