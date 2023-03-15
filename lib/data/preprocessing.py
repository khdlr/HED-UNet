import yaml
import xarray
import rasterio as rio
from rasterio.features import rasterize
import rioxarray
import numpy as np
from PIL import Image
from einops import rearrange
from torch.utils._pytree import tree_map
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from itertools import chain
from multiprocessing import Pool

def tqdm_pmap(fun, args):
  with Pool(8) as pool:
    return list(tqdm(pool.imap_unordered(fun, args), total=len(args)))


def check_exists(cache_dir, name):
  out_path = cache_dir / name
  return out_path.exists()


def save_h5(cache_dir, data, name):
  opts = dict(zlib=True, shuffle=True, complevel=1)
  for var in data.data_vars:
    data[var].encoding.update(opts)
  out_path = cache_dir / name
  out_path.parent.mkdir(parents=True, exist_ok=True)
  data.to_netcdf(out_path, engine='h5netcdf')


def build_sar_scene(args):
  sar_path, cache_dir = args
  mode = sar_path.parent.name
  out_path = f'{mode}/SAR_{Path(sar_path).stem}.nc'

  if check_exists(cache_dir, out_path):
    return

  sar = np.asarray(Image.open(sar_path))
  if sar.ndim == 3:
    sar = sar[..., 0]
  sar = sar.astype(np.float32) / np.float32(255)
  sar = rearrange(sar, 'H W -> 1 H W')
  sar = xarray.DataArray(sar, dims=['channel', 'y', 'x'])
  out = {'SAR': sar}

  mask_path = Path(str(sar_path).replace('sar_images', 'masks').replace('.png', '_mask.png'))
  if mask_path.exists():
    mask = np.asarray(Image.open(mask_path)) > 0
    out['Mask'] = xarray.DataArray(mask, dims=['y', 'x'])

  zone_path = Path(str(sar_path).replace('sar_images', 'zones').replace('.png', '_zones.png'))
  if zone_path.exists():
    zone = np.asarray(Image.open(zone_path))
    zone = np.clip(zone // 63, 0, 3)  # Convert [0,64,128,255] to classes 0,1,2,3
    out['Zones'] = xarray.DataArray(zone, dims=['y', 'x'])

  fronts_path = Path(str(sar_path).replace('sar_images', 'fronts').replace('.png', '_front.png'))
  if fronts_path.exists():
    fronts = np.asarray(Image.open(fronts_path)) > 0
    out['Fronts'] = xarray.DataArray(fronts, dims=['y', 'x'])

  try:
    dataset = xarray.Dataset(out)
  except ValueError:
    print(f'Weird stuff going on with {sar_path}:', tree_map(lambda x: x.shape, out))
    return
  save_h5(cache_dir, dataset, out_path)


def build_s2_scene(args):
  first_img, test_ids, cache_dir = args

  glacier_id = first_img.parent.parent.parent.name
  is_test = glacier_id in test_ids
  mode = 'test' if is_test else 'train'
  tag = Path(first_img.parent).name
  out_path = f'{mode}/S2_{glacier_id}_{tag}.nc'
  if check_exists(cache_dir, out_path):
    return

  bands = []
  for i, band in enumerate(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B11', 'B12'], 1):
    data = rioxarray.open_rasterio(str(first_img).replace('B1', band))
    data.coords['band'] = [i]
    bands.append((data.astype(np.float32) / np.float32(10000)).astype(np.float32))

  img = xarray.concat(bands, 'band')
  out = {}
  out['Sentinel2'] = img
  vals = out['Sentinel2'].values

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

  shapefiles = list(first_img.parent.glob('*.shp'))
  assert len(shapefiles) + 1 == len(out), f'Unused Shapefiles: {shapefiles}'

  dataset = xarray.Dataset(out)
  save_h5(cache_dir, dataset, out_path)


def build_ls_scene(args):
  first_img, test_ids, cache_dir = args

  glacier_id = first_img.parent.parent.parent.name
  is_test = glacier_id in test_ids
  mode = 'test' if is_test else 'train'
  tag = Path(first_img.parent).name
  platform = first_img.stem.split('_')[0]
  platform_id = platform[-1]
  if platform_id in ['4', '5']:
    platform_id = '45'
  out_path = f'{mode}/LS{platform_id}_{glacier_id}_{tag}.nc'

  if check_exists(cache_dir, out_path):
    return

  if platform_id == '8':
    band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B10', 'B11']
    scale = 10000
  elif platform_id == '7':
    band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7', 'B8']
    scale = 255
  elif platform_id == '45':
    band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    scale = 255
  else:
    raise ValueError(f'Unknown Landsat Platform {platform!r}')

  bands = []
  for i, band in enumerate(band_names, 1):
    data = rioxarray.open_rasterio(str(first_img).replace('B1', band))
    data.coords['band'] = [i]
    bands.append((data / scale).astype(np.float32))

  img = xarray.concat(bands, 'band')
  out = {}
  out[f'Landsat{platform_id}'] = img

  profile = rio.open(first_img).profile
  crs, transform = profile['crs'], profile['transform']

  shapefiles = list(first_img.parent.glob('*.shp'))
  shapefiles = set(s for s in shapefiles if not s.stem.endswith('domain'))
  for mask_path in first_img.parent.glob('*_MASK.shp'):
    shapefiles.discard(mask_path)
    try:
      geom = gpd.read_file(mask_path).to_crs(crs)
    except ValueError:
      print(f"Couldn't reproject mask for {first_img}")
      return
    geom = geom[geom.DN > 0.5].geometry
    front = rasterize(geom, out_shape=img.shape[1:], default_value=1, transform=transform)
    xr = xarray.DataArray(front, coords=[img.y, img.x])
    xr = xr.rio.write_transform(transform).rio.write_crs(crs)
    out['Mask'] = xr

  for termpicks in first_img.parent.glob('*_Termpicks.shp'):
    shapefiles.discard(termpicks)
    geom = gpd.read_file(termpicks).to_crs(crs).geometry
    front = rasterize(geom, all_touched=True, out_shape=img.shape[1:], default_value=1, transform=transform)
    xr = xarray.DataArray(front, coords=[img.y, img.x])
    xr = xr.rio.write_transform(transform).rio.write_crs(crs)
    out['Termpicks'] = xr

  for kochtitzky in first_img.parent.glob('*_Kochtitzky.shp'):
    shapefiles.discard(kochtitzky)
    geom = gpd.read_file(kochtitzky).to_crs(crs).geometry
    front = rasterize(geom, all_touched=True, out_shape=img.shape[1:], default_value=1, transform=transform)
    xr = xarray.DataArray(front, coords=[img.y, img.x])
    xr = xr.rio.write_transform(transform).rio.write_crs(crs)
    out['Kochtitzky'] = xr

  assert not shapefiles, f'Unused Shapefiles: {shapefiles}'
  assert len(out) > 1, f'Image without Labels: {first_img}'

  dataset = xarray.Dataset(out)
  save_h5(cache_dir, dataset, out_path)


def build_tud_scene(args):
  first_img, test_ids, cache_dir = args

  glacier_id = first_img.parent.parent.name
  _, glacier_name = glacier_id.split('.')
  is_test = glacier_id in test_ids
  mode = 'test' if is_test else 'train'
  tag = Path(first_img.parent).name
  out_path = f'{mode}/LS8_TUD_{glacier_name}_{tag}.nc'

  if check_exists(cache_dir, out_path):
    return

  band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B10', 'B11']
  scale = 255

  bands = []
  for i, band in enumerate(band_names, 1):
    data = rioxarray.open_rasterio(str(first_img).replace('B1', band))
    data.coords['band'] = [i]
    bands.append((data / scale).astype(np.float32))

  img = xarray.concat(bands, 'band')
  out = {}
  out[f'Landsat8_Normalized'] = img

  profile = rio.open(first_img).profile
  crs, transform = profile['crs'], profile['transform']

  full_name = glacier_name
  if glacier_name[-2] == '_':
    # Drop _a, _b, etc.
    full_name = glacier_name[:-2]
  if glacier_name in ('zachariae_isstrom_north', 'zachariae_isstrom_south'):
    full_name = 'zachariae_isstrom'

  polygon_root = Path(str(first_img.parent).replace(glacier_id, full_name).replace('reference_data_geotif', 'polygons'))
  polygon_path, = polygon_root.glob('*.shp')
  try:
    geom = gpd.read_file(polygon_path).to_crs(crs).geometry
  except ValueError:
    print(f"Couldn't reproject mask for {first_img}")
    return
  mask = rasterize(geom, out_shape=img.shape[1:], default_value=1, transform=transform)
  mask = 1 - mask
  xr = xarray.DataArray(mask, coords=[img.y, img.x])
  xr = xr.rio.write_transform(transform).rio.write_crs(crs)
  out['Mask'] = xr

  lines_root = Path(str(first_img.parent).replace(glacier_id, full_name).replace('reference_data_geotif', 'lines'))
  lines_path, = lines_root.glob('*.shp')
  try:
    geom = gpd.read_file(lines_path).to_crs(crs).geometry
  except ValueError:
    print(f"Couldn't reproject coastline for {first_img}")
    return
  front = rasterize(geom, out_shape=img.shape[1:], default_value=1, transform=transform)
  xr = xarray.DataArray(front, coords=[img.y, img.x])
  xr = xr.rio.write_transform(transform).rio.write_crs(crs)
  out['TUD_Fronts'] = xr

  dataset = xarray.Dataset(out)
  save_h5(cache_dir, dataset, out_path)


class Preprocessing:
  def __init__(self, data_root):
    self.root = Path(data_root)
    self.cache = self.root / 'cubes'

  def build_scenes(self):
    print('Preprocessing SAR images')
    sar_images = chain(
        self.root.glob(f'SARImage/Gourmelon/sar_images/*/*.png'),
        self.root.glob(f'SARImage/CALFIN_Sentinel1/sar_images/*/*.png')
    )
    tqdm_pmap(build_sar_scene, [(img, self.cache) for img in sar_images])

    split = yaml.safe_load(open('split.yml'))
    test_ids = split['GrIS_Test'] + split['GIC_Test'] + split['TUD_Test']

    print('Preprocessing Sentinel 2 images')
    first_imgs = self.root.glob(f'OpticalImage/*/*_tif/*/Sentinel2/*/*B1.tif')
    tqdm_pmap(build_s2_scene, [(img, test_ids, self.cache) for img in first_imgs])

    print('Preprocessing Landsat images')
    first_imgs = self.root.glob(f'OpticalImage/*/*_tif/*/Landsat/*/*B1.tif')
    tqdm_pmap(build_ls_scene, [(img, test_ids, self.cache) for img in first_imgs])

    print('Preprocessing TUD images')
    first_imgs = self.root.glob(f'TUD/reference_data_geotif/*/*/L8_B1.tif')
    tqdm_pmap(build_tud_scene, [(img, test_ids, self.cache) for img in first_imgs])


if __name__ == '__main__':
  config = yaml.safe_load(open('config.yml'))
  prep = Preprocessing(config['data_root'])
  prep.build_scenes()
