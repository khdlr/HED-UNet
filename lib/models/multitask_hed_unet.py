import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from functools import partial
from .layers import Convx2, DownBlock, UpBlock, WithSE, PreactConvx2
from einops import rearrange


class NoBatchConv(nn.Conv2d):
  def forward(self, inputs):
    x = inputs.unsqueeze(0)
    x = super().forward(x)
    return x.squeeze(0)


class Interpolator:
  def __init__(self, output_size):
    self.output_size = output_size

  def __call__(self, x):
    x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
    return x


class MultitaskHEDUNet(nn.Module):
    """
    A straight-forward HED-UNet implementation
    """

    def __init__(self, input_spec, output_spec, base_channels=16,
                 conv_block=Convx2, padding_mode='replicate', batch_norm=False,
                 squeeze_excitation=False, merging='attention', stack_height=5,
                 deep_supervision=True):
        super().__init__()
        bc = base_channels
        if squeeze_excitation:
            conv_block = WithSE(conv_block)

        self.init = nn.ModuleDict()
        for kind, channels in input_spec.items():
          self.init[kind] = NoBatchConv(channels, bc, 1)

        self.predictors = nn.ModuleList([
          nn.Conv2d((1<<i)*bc, 4, 1)
          for i in reversed(range(stack_height + 1))
        ])
        self.queries = nn.ModuleList([
          nn.Conv2d((1<<i)*bc, 1, 1, bias=False)
          for i in reversed(range(stack_height + 1))
        ])

        conv_args = dict(
            conv_block=conv_block,
            bn=batch_norm,
            padding_mode=padding_mode
        )

        self.down_blocks = nn.ModuleList([
            DownBlock((1<<i)*bc, (2<<i)*bc, **conv_args)
            for i in range(stack_height)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock((2<<i)*bc, (1<<i)*bc, **conv_args)
            for i in reversed(range(stack_height))
        ])

        self.deep_supervision = deep_supervision

    def extract_predictions(self, seg):
        out = {}
        out['Zones'] = seg

        logprobs = torch.log_softmax(seg[:, 1:], dim=1)
        logprobs_w1 = F.max_pool2d(logprobs, 3, padding=1)
        logprobs_w2 = F.max_pool2d(logprobs_w1, 3, padding=1)

        rock_w, glacier_w, ocean_w = torch.split(logprobs, [1,1,1], dim=1)
        edge = torch.minimum(glacier_w, ocean_w)

        out['Fronts'] = edge
        out['Kochtitzky'] = edge
        out['Termpicks'] = edge

        rock, glacier, ocean = torch.split(logprobs, [1,1,1], dim=1)

        foreground = torch.logaddexp(rock, glacier)
        out['Mask'] =  torch.cat([ocean, foreground], dim=1)
        return out

    def forward(self, inputs):
        xs = []
        for element in inputs:
          assert len(element) == 1
          kind, = element.keys()
          x, = element.values()
          xs.append(self.init[kind](x))
        x = torch.stack(xs, dim=0)
        B, _, H, W = x.shape

        skip_connections = []
        for block in self.down_blocks:
            skip_connections.append(x)
            x = block(x)

        multilevel_features = [x]
        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)
            multilevel_features.append(x)

        scale = Interpolator(output_size=(H, W))
        deep_outputs = []
        predictions = []
        queries = []
        levels = zip(multilevel_features, self.predictors, self.queries)
        for i, (feature_map, predictor, query) in enumerate(levels):
          prediction = predictor(feature_map)
          q = query(feature_map)
          deep_outputs.append(self.extract_predictions(prediction))
          predictions.append(scale(prediction))
          queries.append(scale(q))

        D = 0
        predictions = torch.stack(predictions, dim=D)
        attn = F.softmax(torch.stack(queries, dim=D), dim=D)[:, :, 0]

        raw_output = torch.einsum('rbchw,rbhw->bchw', predictions, attn)
        outputs = self.extract_predictions(raw_output)

        if self.deep_supervision:
            return outputs, deep_outputs
        else:
            return outputs
