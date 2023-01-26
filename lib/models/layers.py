import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PreactConvx2(nn.Module):
    def __init__(self, c_in, c_out, bn, padding_mode='zeros'):
        super().__init__()
        conv_args = dict(padding=1, padding_mode=padding_mode, bias=not bn)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, **conv_args)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, **conv_args)
        if bn:
            self.bn1 = nn.BatchNorm2d(c_in)
            self.bn2 = nn.BatchNorm2d(c_out)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))
        return x

class Convx2(nn.Module):
    def __init__(self, c_in, c_out, bn, padding_mode='zeros'):
        super().__init__()
        conv_args = dict(padding=1, padding_mode=padding_mode, bias=not bn)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, **conv_args)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, **conv_args)
        if bn:
            self.bn1 = nn.BatchNorm2d(c_out)
            self.bn2 = nn.BatchNorm2d(c_out)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, conv_block=Convx2, batch_norm=True):
        super().__init__()
        if c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1)
        else:
            self.skip = Identity()

        self.convblock = conv_block(c_in, c_out, batch_norm)

    def forward(self, x):
        skipped = self.skip(x)
        residual = self.convblock(x)
        return skipped + residual


class DenseBlock(nn.Module):
    def __init__(self, c_in, c_out, bn, dense_size=8):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, bias=not bn)
        self.dense_convs = nn.ModuleList([
            nn.Conv2d(c_in + i * dense_size, dense_size, **conv_args)
            for i in range(4)
        ])
        self.final = nn.Conv2d(c_in + 4 * dense_size, c_out, **conv_args)

        if bn:
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(dense_size)
                for i in range(4)
            ])
            self.bn_final = nn.BatchNorm2d(c_out)
        else:
            self.bns = nn.ModuleList([Identity() for i in range(4)])
            self.bn_final = Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv, bn in zip(self.dense_convs, self.bns):
            x = torch.cat([x, self.relu(bn(conv(x)))], dim=1)
        x = self.relu(self.bn_final(self.final(x)))
        return x


class SqueezeExcitation(nn.Module):
    """
    adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    See: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = int(math.ceil(channels / reduction))
        self.squeeze = nn.Conv2d(channels, reduced, 1)
        self.excite = nn.Conv2d(reduced, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.avg_pool2d(x, x.shape[2:])
        y = self.relu(self.squeeze(y))
        y = torch.sigmoid(self.excite(y))
        return x * y


def WithSE(conv_block, reduction=8):
    def make_block(c_in, c_out, **kwargs):
        return nn.Sequential(
            conv_block(c_in, c_out, **kwargs),
            SqueezeExcitation(c_out, reduction=reduction)
        )
    make_block.__name__ = f"WithSE({conv_block.__name__})"
    return make_block


class DownBlock(nn.Module):
    """
    UNet Downsampling Block
    """
    def __init__(self, c_in, c_out, conv_block=Convx2,
                 bn=True, padding_mode='zeros'):
        super().__init__()
        bias = not bn
        self.convdown = nn.Conv2d(c_in, c_in, 2, stride=2, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(c_in)
        else:
            self.bn = Identity()
        self.relu = nn.ReLU(inplace=True)

        self.conv_block = conv_block(c_in, c_out, bn=bn, padding_mode=padding_mode)

    def forward(self, x):
        x = self.relu(self.bn(self.convdown(x)))
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """
    UNet Upsampling Block
    """
    def __init__(self, c_in, c_out, conv_block=Convx2,
                 bn=True, padding_mode='zeros'):
        super().__init__()
        bias = not bn
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(c_in // 2)
        else:
            self.bn = Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = conv_block(c_in, c_out, bn=bn, padding_mode=padding_mode)

    def forward(self, x, skip):
        x = self.relu(self.bn(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x
