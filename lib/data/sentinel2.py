import numpy as np
from abc import abstractmethod
import rasterio as rio
from rasterio.features import rasterize
import geopandas as gpd
import yaml
from typing import Iterable

from .base import GlacierFrontDataset

class SentinelMixin:
  sensor = 'Sentinel2'

  def iter_bands(self, band1):
    for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B11', 'B12']:
      yield str(band1).replace('B1', band)

class LandsatMixin:
  sensor = 'Landsat'

  def iter_bands(self, band1):
    for band in ['TODO']:
      yield str(band1).replace('B1', band)

class GreenlandMixin:
  location = 'GreenlandIceSheet'

class MountainMixin:
  location = 'MountainRange'

class OpticalDataset(GlacierFrontDataset):
  location: str
  sensor: str

  def __post_init__(self):
    split = yaml.safe_load((self.root / 'split.yml').open())
    loc = 'GrIS' if self.location == 'GreenlandIceSheet' else 'GIC'
    self.glaciers = split[f'{loc}_{self.mode.title()}']

  def cache_tag(self) -> str:
    return self.location + self.sensor

  def generate_images(self):
    for glacier in self.glaciers:
      for first_img in self.root.glob(f'OpticalImage/{self.location}/*_tif/{glacier}/{self.sensor}/*/*B1.tif'):
        folder = first_img.parent
        print(folder)
        imagery = np.stack([rio.open(band).read(1) for band in self.iter_bands(first_img)])
        out = {self.sensor: imagery}

        for mask_path in folder.glob('*_MASK.tif'):
          with rio.open(mask_path) as raster:
            out['Mask'] = raster.read(1)

        for termpicks in folder.glob('*_Termpicks.shp'):
          gdf = gpd.read_file(termpicks)
          profile = rio.open(first_img).profile

          print(profile)

          front = rasterize(
              gdf.to_crs(profile['crs']).geometry,
              all_touched=True,
              out_shape=imagery.shape[1:],
              default_value=1,
              transform=profile['transform'])
          out['Termpicks'] = front

        yield out


class GreenlandS2Dataset(OpticalDataset, SentinelMixin, GreenlandMixin):
  ...

if __name__ == "__main__":
    config = {
      'data_root': "/home/user/_data/FromTian/",
      'tilesize': 512
    }
    ds = GreenlandS2Dataset(config, mode='train')
    print('Tiles:', len(ds))
    for key, val in ds[0].items():
        print(key, val.shape, val.dtype)
