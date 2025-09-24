from __future__ import annotations
import torch
import torch.nn.functional as F
import abc
from typing import cast, TYPE_CHECKING


from CoBiSyn.data.common import ForwardData, ForwardBatch

if TYPE_CHECKING:
    from typing import Dict, List
    from collections.abc import Sequence, Callable, Mapping


def collate_scalar(scalars: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    return torch.stack(scalars, dim=0)

def collate_1d_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=(0, max_size - f.size(-1)), mode='constant', value=0)
        for f in features
    ]
    return torch.stack(features_padded, dim=0)

def collate_2d_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=(0, max_size - f.size(-1), 0, max_size - f.size(-2)), mode='constant', value=0)
        for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_1d_features(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=(0, 0, 0, max_size-f.size(-2)), mode='constant', value=0) 
        for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_padding_masks(masks: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    masks_padded = [
        F.pad(m.bool(), pad=(0, max_size - m.size(-1)), mode='constant', value=False)
        for m in masks
    ]
    return torch.stack(masks_padded, dim=0)


def apply_collate(
    funcs: Mapping[str, Callable[[Sequence[torch.Tensor], int], torch.Tensor]],
    data_list: Sequence[Dict[str, torch.Tensor]],
    max_size: int
) -> Dict[str, torch.Tensor]:
    transpose = {k: [d[k] for d in data_list] for k in funcs.keys()}
    batch = {k: funcs[k](transpose[k], max_size) for k in funcs.keys()}
    return batch


class Collater(abc.ABC):
    def __init__(self, max_num_atoms: int = 96, max_num_tokens: int = 8):
        super().__init__()
        self._max_num_atoms = max_num_atoms
        self._max_num_tokens = max_num_tokens

    @abc.abstractmethod
    def __call__(self, data_list): ...
    

class ForwardCollater(Collater):
    def __init__(self, max_num_atoms = 96, max_num_tokens = 8):
        super().__init__(max_num_atoms, max_num_tokens)
        self._funcs_mol = {
            "cond_atoms": collate_1d_tokens,
            "cond_bonds": collate_2d_tokens,
            "cond_atoms_padding_mask": collate_padding_masks,
        }
        self._funcs_token = {
            "token_types": collate_1d_tokens,
            "token_padding_mask": collate_padding_masks,
            "rxn_indices": collate_1d_tokens,
            "reactant_fps": collate_1d_features,
        }

    def __call__(self, data_list: Sequence[ForwardData]) -> ForwardBatch:
        data_list_t = cast(List[Dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self._funcs_mol, data_list_t, self._max_num_atoms),
            **apply_collate(self._funcs_token, data_list, self._max_num_tokens)
        }
        return cast(ForwardBatch, batch)
