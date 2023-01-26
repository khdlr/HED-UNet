import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Convx2, DownBlock, UpBlock, WithSE, PreactConvx2


class HEDUNet(nn.Module):
    """
    A straight-forward HED-UNet implementation
    """

    def __init__(self, input_channels, output_channels=2, base_channels=16,
                 conv_block=Convx2, padding_mode='replicate', batch_norm=False,
                 squeeze_excitation=False, merging='attention', stack_height=5,
                 deep_supervision=True):
        super().__init__()
        bc = base_channels
        if squeeze_excitation:
            conv_block = WithSE(conv_block)
        self.init = nn.Conv2d(input_channels, bc, 1)

        self.output_channels = output_channels

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
            nn.Conv2d((1<<i)*bc, output_channels, 1)
            for i in reversed(range(stack_height + 1))
        ])

        self.deep_supervision = deep_supervision
        self.merging = merging
        if merging == 'attention':
            self.queries = nn.ModuleList([
                nn.Conv2d((1<<i)*bc, output_channels, 1)
                for i in reversed(range(stack_height + 1))
            ])
        elif merging == 'learned':
            self.merge_predictions = nn.Conv2d(output_channels*(stack_height+1), output_channels, 1)
        else:
            # no merging
            pass


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
            full_scale_preds.append(F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=True))

        predictions = torch.cat(full_scale_preds, dim=1)

        if self.merging == 'attention':
            queries = [F.interpolate(q(feat), size=(H, W), mode='bilinear', align_corners=True)
                    for q, feat in zip(self.queries, multilevel_features)]
            queries = torch.cat(queries, dim=1)
            queries = queries.reshape(B, -1, self.output_channels, H, W)
            attn = F.softmax(queries, dim=1)
            predictions = predictions.reshape(B, -1, self.output_channels, H, W)
            combined_prediction = torch.sum(attn * predictions, dim=1)
        elif self.merging == 'learned':
            combined_prediction = self.merge_predictions(predictions)
        else:
            combined_prediction = predictions_list[-1]

        if self.deep_supervision:
            return combined_prediction, list(reversed(predictions_list))
        else:
            return combined_prediction
