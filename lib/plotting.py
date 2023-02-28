from matplotlib.colors import LinearSegmentedColormap
import torch
import numpy as np
import wandb
from einops import rearrange
from skimage.transform import downscale_local_mean

FLATUI = {'Turquoise': (0.10196078431372549, 0.7372549019607844, 0.611764705882353),
 'Emerald': (0.1803921568627451, 0.8, 0.44313725490196076),
 'Peter River': (0.20392156862745098, 0.596078431372549, 0.8588235294117647),
 'Amethyst': (0.6078431372549019, 0.34901960784313724, 0.7137254901960784),
 'Wet Asphalt': (0.20392156862745098, 0.28627450980392155, 0.3686274509803922),
 'Green Sea': (0.08627450980392157, 0.6274509803921569, 0.5215686274509804),
 'Nephritis': (0.15294117647058825, 0.6823529411764706, 0.3764705882352941),
 'Belize Hole': (0.1607843137254902, 0.5019607843137255, 0.7254901960784313),
 'Wisteria': (0.5568627450980392, 0.26666666666666666, 0.6784313725490196),
 'Midnight Blue': (0.17254901960784313, 0.24313725490196078, 0.3137254901960784),
 'Sun Flower': (0.9450980392156862, 0.7686274509803922, 0.058823529411764705),
 'Carrot': (0.9019607843137255, 0.49411764705882355, 0.13333333333333333),
 'Alizarin': (0.9058823529411765, 0.2980392156862745, 0.23529411764705882),
 'Clouds': (0.9254901960784314, 0.9411764705882353, 0.9450980392156862),
 'Concrete': (0.5843137254901961, 0.6470588235294118, 0.6509803921568628),
 'Orange': (0.9529411764705882, 0.611764705882353, 0.07058823529411765),
 'Pumpkin': (0.8274509803921568, 0.32941176470588235, 0.0),
 'Pomegranate': (0.7529411764705882, 0.2235294117647059, 0.16862745098039217),
 'Silver': (0.7411764705882353, 0.7647058823529411, 0.7803921568627451),
 'Asbestos': (0.4980392156862745, 0.5490196078431373, 0.5529411764705883)}

FLATUI_U8 = {k: (np.asarray(v) * 255).astype(np.uint8) for k, v in FLATUI.items()}

def flatui_cmap(*colors):
    ts = np.linspace(0, 1, len(colors))

    segmentdata = dict(
        red=[[t, FLATUI[col][0], FLATUI[col][0]] for col, t in zip(colors, ts)],
        green=[[t, FLATUI[col][1], FLATUI[col][1]] for col, t in zip(colors, ts)],
        blue=[[t, FLATUI[col][2], FLATUI[col][2]] for col, t in zip(colors, ts)],
    )

    return LinearSegmentedColormap('flatui', segmentdata=segmentdata, N=256)


def to_rgb(ary, kind):
  is_pred = kind.startswith('Pred')
  kind = kind.removeprefix('Pred')
  if kind in ['Kochtitzky', 'Termpicks', 'Fronts', 'Mask']:
    if is_pred:
      if kind == 'Mask':
        ary = torch.softmax(torch.from_numpy(ary), dim=1)[1].numpy()
      else:
        ary = 1 / (1 + np.exp(-ary))
    if ary.shape[0] == 1:
      ary = ary[0]
    if ary.ndim == 2:
      ary = ary[..., np.newaxis]
    if kind == 'Mask':
      zero = np.asarray(FLATUI['Green Sea'])
      one = np.asarray(FLATUI['Carrot'])
    else:
      zero = np.asarray(FLATUI['Clouds'])
      one = np.asarray(FLATUI['Midnight Blue'])
    rgb = ary * one + (1-ary) * zero
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb
  elif kind == 'Zones':
    if ary.ndim == 3:
      ary = ary.argmax(axis=0)
    colors = np.asarray([
      FLATUI_U8['Midnight Blue'],  # NA
      FLATUI_U8['Wisteria'],       # Rock
      FLATUI_U8['Clouds'],         # Glacier
      FLATUI_U8['Green Sea']       # Ocean
    ])
    out = colors[ary]
    return out
  elif kind in ['Landsat45', 'Landsat7', 'Landsat8', 'Sentinel2']:
    rgb = rearrange(ary[[3,2,1]], 'C H W -> H W C')
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb
  elif kind == 'SAR':
    gray = (np.clip(ary[0], 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb
  else:
    raise ValueError(f'Unsupported Kind: {kind!r}')


def log_image(patches, metadata, tag, step):
  out_types = set()
  # code.interact(locals=locals())

  for p, m in zip(patches, metadata):
    y0, y1 = m['py0'], m['py1']
    x0, x1 = m['px0'], m['px1']
    for k, ary in p.items():
      *_, H, W = ary.shape
      p[k] = ary[..., y0:H-y1, x0:W-x1]
      out_types.add(k)

  y_max = max(d['y1'] for d in metadata)
  x_max = max(d['x1'] for d in metadata)
  y_min = min(d['y0'] for d in metadata)
  x_min = min(d['x0'] for d in metadata)
  H, W = y_max - y_min, x_max - x_min

  imgs = {k: np.zeros([H, W, 3], dtype=np.uint8) for k in out_types}
  for p, m in zip(patches, metadata):
    y0, y1 = m['y0'] - y_min, m['y1'] - y_min
    x0, x1 = m['x0'] - x_min, m['x1'] - x_min
    for k in imgs:
      imgs[k][y0:y1,x0:x1] = to_rgb(p[k], k)
  
  row_1a = []
  row_2a = []
  row_1b = []
  row_2b = []
  for k in list(imgs):
    if k.startswith('Pred'):
      stripped = k.removeprefix('Pred')
      if stripped not in imgs:
        imgs[stripped] = np.ones_like(imgs[k]) * 255

  for k in sorted(imgs):
    if k in ['Landsat45', 'Landsat7', 'Landsat8', 'Sentinel2', 'SAR']:
      row_1a.append(imgs[k])
      row_2a.append(np.ones_like(imgs[k]) * 255)
    else:
      if k.startswith('Pred'):
        row_2b.append(imgs[k])
      else:
        row_1b.append(imgs[k])

  img = np.concatenate([
    np.concatenate(row_1a + row_1b, axis=1),
    np.concatenate(row_2a + row_2b, axis=1),
  ], axis=0)

  if 'SAR' in imgs:
    img = downscale_local_mean(img, (6, 6, 1))
  else:
    img = downscale_local_mean(img, (2, 2, 1))

  wandb.log({tag: wandb.Image(img)}, step=step)

