import yaml
import xarray
import rasterio as rio
from rasterio.features import rasterize
import rioxarray
import numpy as np
from PIL import Image
from einops import rearrange
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

class Preprocessing:
  def __init__(self, data_root):
    self.root = Path(data_root)
    self.cache = self.root / 'cubes'

  def save_h5(self, data, name):
    opts = dict(zlib=True, shuffle=True, complevel=1)
    for var in data.data_vars:
      data[var].encoding.update(opts)
    out_path = self.cache / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(out_path, engine='h5netcdf')

  def build_sar(self):
    for sar_path in tqdm(list(self.root.glob(f'SARImage/sar_images/*/*.png'))):
      sar = np.asarray(Image.open(sar_path))
      sar = rearrange(sar, 'H W -> 1 H W')
      sar = xarray.DataArray(sar, dims=['channel', 'y', 'x'])
      out = {'SAR': sar}

      mask_path = Path(str(sar_path).replace('sar_images', 'mask').replace('.png', '_mask.png'))
      if mask_path.exists():
        mask = np.asarray(Image.open(mask_path)) > 0
        out['Mask'] = xarray.DataArray(mask, dims=['y', 'x'])

      zone_path = Path(str(img_path).replace('sar_images', 'zones').replace('.png', '_zones.png'))
      if zone_path.exists:
        zone = np.asarray(Image.open(zone_path))
        zone = np.clip(zone // 63, 0, 3)  # Convert [0,64,128,255] to classes 0,1,2,3
        out['Zones'] = xarray.DataArray(zone, dims=['y', 'x'])

      fronts_path = Path(str(img_path).replace('sar_images', 'fronts').replace('.png', '_fronts.png'))
      if fronts_path.exists():
        fronts = np.asarray(Image.open(mask_path)) > 0
        out['fronts'] = xarray.DataArray(fronts, dims=['y', 'x'])
         
      dataset = xarray.Dataset(out)
      mode = mask_path.parent.name
      self.save_h5(dataset, f'{mode}/SAR_{Path(img_path).stem}.nc')

  def build_s2(self):
    split = yaml.safe_load((self.root / 'split.yml').open())
    test_ids = split['GrIS_Test'] + split['GIC_Test']

    for first_img in self.root.glob(f'OpticalImage/*/*_tif/*/Sentinel2/*/*B1.tif'):
      glacier_id = first_img.parent.parent.parent.name
      print(glacier_id)
      is_test = glacier_id in test_ids

      bands = []
      for i, band in enumerate(['B1', 'B2', 'B3', 'B4', 'B5', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B11', 'B12'], 1):
        data = rioxarray.open_rasterio(str(first_img).replace('B1', band))
        data.coords['band'] = [i]
        bands.append(data)

      img = xarray.concat(bands, 'band')
      out = {}
      out['Sentinel2'] = img

      profile = rio.open(first_img).profile
      crs, transform = profile['crs'], profile['transform']
      for mask_path in first_img.parent.glob('*_MASK.shp'):
        geom = gpd.read_file(mask_path).to_crs(crs)
        geom = geom[geom.DN > 0.5].geometry
        front = rasterize(geom, out_shape=img.shape[1:], default_value=1, transform=transform)
        xr = xarray.DataArray(front, coords=[img.y, img.x])
        xr = xr.rio.write_transform(transform).rio.write_crs(crs)
        out['Mask'] = xr

      for termpicks in first_img.parent.glob('*_Termpicks.shp'):
        geom = gpd.read_file(termpicks).to_crs(crs).geometry
        front = rasterize(geom, all_touched=True, out_shape=img.shape[1:], default_value=1, transform=transform)
        xr = xarray.DataArray(front, coords=[img.y, img.x])
        xr = xr.rio.write_transform(transform).rio.write_crs(crs)
        out['Termpicks'] = xr

      dataset = xarray.Dataset(out)
      mode = 'test' if is_test else 'train'
      self.save_h5(dataset, f'{mode}/S2_{glacier_id}_{Path(first_img.parent).name}.nc')


if __name__ == '__main__':
  prep = Preprocessing('/hdd/phd/FromTian')
  prep.build_sar()
