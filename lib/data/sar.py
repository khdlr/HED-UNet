import numpy as np
from PIL import Image
from einops import rearrange

from .base import GlacierFrontDataset


class SARDataset(GlacierFrontDataset):
  def __post_init__(self):
    ...

  def cache_tag(self):
    return 'SAR'

  def generate_images(self):
    for mask_path in self.root.glob(f'SARImage/fronts/{self.mode}/*.png'):
      mask = np.asarray(Image.open(mask_path)) > 0

      img_path = str(mask_path).replace('fronts', 'sar_images').replace('_front', '')
      img = np.asarray(Image.open(img_path))
      img = rearrange(img, 'H W -> 1 H W')

      zone_path = str(mask_path).replace('fronts', 'zones').replace('_front', '_zones')
      zone = np.asarray(Image.open(zone_path))
      zone = np.clip(zone // 63, 0, 3)  # Convert [0,64,128,255] to classes 0,1,2,3

      yield {'SAR': img, 'Fronts': mask, 'Zones': zone}


if __name__ == "__main__":
    config = {
      'data_root': "/home/user/_data/FromTian/",
      'tilesize': 512
    }
    ds = SARDataset(config, mode='train')
    print('Tiles:', len(ds))
    for key, val in ds[0].items():
        print(key, val.shape, val.dtype)
