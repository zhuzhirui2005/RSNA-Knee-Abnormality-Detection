import torch
from .vnet import VNet
def load_segmenter(weights, device):
    model = VNet(spatial_dims=2, in_channels=1, out_channels=4,
                 act=('elu', {'inplace': True}), dropout_prob=0., dropout_dim=2)
    state = torch.load(weights, map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict({k.removeprefix('model.net.'): v for k, v in state.items()}, strict=True)
    return model.to(device).eval().requires_grad_(False)
