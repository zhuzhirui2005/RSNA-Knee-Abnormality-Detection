"""Torch equivalents of the MONAI 0.6 layers used by the released 2-D VNet."""

from torch import nn


class _Factory:
    BATCH = "batch"
    CONV = "conv"
    CONVTRANS = "convtrans"
    DROPOUT = "dropout"

    def __init__(self, kind):
        self.kind = kind

    def __getitem__(self, key):
        name, dim = key if isinstance(key, tuple) else (key, 2)
        if self.kind == "act":
            return {"elu": nn.ELU, "prelu": nn.PReLU}[name]
        if self.kind == "norm":
            return {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dim]
        if self.kind == "dropout":
            return {1: nn.Dropout, 2: nn.Dropout2d, 3: nn.Dropout3d}[dim]
        return (
            {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
            if name == "convtrans"
            else {2: nn.Conv2d, 3: nn.Conv3d}
        )[dim]


Act, Conv, Dropout, Norm = [_Factory(k) for k in ["act", "conv", "dropout", "norm"]]


def split_args(value):
    return value if isinstance(value, tuple) else (value, {})


class Convolution(nn.Sequential):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size, act, norm):
        super().__init__()
        self.add_module(
            "conv",
            Conv["conv", dimensions](
                in_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
        )
        adn = nn.Sequential()
        adn.add_module("N", Norm["batch", dimensions](out_channels))
        self.add_module("adn", adn)
