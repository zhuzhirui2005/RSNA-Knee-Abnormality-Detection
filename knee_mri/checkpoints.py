"""Load plain tensor weights and training checkpoints."""
from pathlib import Path
import torch
from .model import KneeClassifier


def load_model(path, crop=352, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved.get('model', saved)
    if 'state_gate' in state and torch.count_nonzero(state['state_gate']):
        raise ValueError('This model requires active report-state heads, which are not part of the released architecture.')
    state = {k: v for k, v in state.items() if not k.startswith(('state_', 'bin_'))}
    model = KneeClassifier('convnext_small', None, crop=crop, spatial=True, extent='ext_u' in state)
    model.load_state_dict(state, strict=True)
    return model.to(device).to(memory_format=torch.channels_last).eval()


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
