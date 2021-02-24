import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat 
from .layers import Convx2, DownBlock, UpBlock, WithSE, PreactConvx2


class HEDUNet(nn.Module):
    """
    A straight-forward UNet implementation
    """
    tasks = ['seg', 'edge']

    def __init__(self, input_channels, classes=20, base_channels=16,
                 conv_block=Convx2, padding_mode='replicate', batch_norm=False,
                 squeeze_excitation=False, stack_height=5,
                 deep_supervision=True):
        super().__init__()
        bc = base_channels
        if squeeze_excitation:
            conv_block = WithSE(conv_block)
        self.init = nn.Conv2d(input_channels, bc, 1)

        self.classes = classes

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

        self.predictors = nn.ModuleList([
            nn.Conv2d((1<<i)*bc, 1+classes, 1)
            for i in reversed(range(stack_height + 1))
        ])

        self.deep_supervision = deep_supervision
        self.queries = nn.ModuleList([
            nn.Conv2d((1<<i)*bc, 2, 1)
            for i in reversed(range(stack_height + 1))
        ])


    def forward(self, x):
        B, _, H, W = x.shape
        x = self.init(x)

        skip_connections = []
        for block in self.down_blocks:
            skip_connections.append(x)
            x = block(x)

        multilevel_features = [x]
        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)
            multilevel_features.append(x)

        predictions_list = []
        full_scale_preds = []
        for feature_map, predictor in zip(multilevel_features, self.predictors):
            prediction = predictor(feature_map)
            predictions_list.append(prediction)
            full_scale_preds.append(F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=False))

        predictions = torch.stack(full_scale_preds, dim=2)
        # B x 1+C x K x H x W

        queries = [F.interpolate(q(feat), size=(H, W), mode='bilinear', align_corners=False)
                for q, feat in zip(self.queries, multilevel_features)]
        queries = torch.stack(queries, dim=2)
        # queries: B x 2 x K x H x W
        B, _, K, H, W = queries.shape
        attns = F.softmax(queries, dim=2)
        edge_attn = attns[:, 0]
        seg_attn  = attns[:, 1]
        full_attn = torch.cat(
            [attns[:, [0]], attns[:, [1]].expand(B, self.classes, K, H, W)],
            dim=1
        )
        combined_prediction = torch.einsum('bckhw,bckhw->bchw', full_attn, predictions)

        if self.deep_supervision:
            return combined_prediction, list(reversed(predictions_list))
        else:
            return combined_prediction
