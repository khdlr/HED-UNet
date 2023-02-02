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

  def __call__(self, inputs):
    x = inputs.unsqueeze(0)
    x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
    return x.squeeze(0)

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

        self.predictors = nn.ModuleDict()
        self.queries = nn.ModuleDict()
        for kind, spec in output_spec.items():
          self.predictors[kind] = nn.ModuleList([
            NoBatchConv((1<<i)*bc, spec['classes'], 1)
            for i in reversed(range(stack_height + 1))
          ])
          self.queries[kind] = nn.ModuleList([
            NoBatchConv((1<<i)*bc, 1, 1, bias=False)
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

    def forward(self, inputs, output_kinds):
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
        # outputs[batch_idx][kind] = ...
        # deep_outputs[batch_idx][kind][level] = ...
        outputs = []
        deep_outputs = []
        for kinds, *features in zip(output_kinds, *multilevel_features):
          output = {}
          deep_output = {}
          for kind in kinds:
            deep_outs = []
            predictions = []
            queries = []
            for feature_map, predictor, query in zip(features, self.predictors[kind], self.queries[kind]):
              prediction = predictor(feature_map)
              q = query(feature_map)
              deep_outs.append(prediction)
              predictions.append(scale(prediction))
              queries.append(scale(q))
            predictions = torch.stack(predictions, dim=0) # Level x Channel x H x W
            attn = F.softmax(torch.cat(queries, dim=0), dim=0) # Level x H x W
            final_output = torch.einsum('rchw,rhw->chw', predictions, attn)

            output[kind] = final_output
            deep_output[kind] = deep_outs
          outputs.append(output)
          deep_outputs.append(deep_output)

        if self.deep_supervision:
            return outputs, deep_outputs
        else:
            return outputs
