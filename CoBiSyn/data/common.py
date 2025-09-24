from __future__ import annotations
import os
import torch
import enum
from typing import TypedDict, TYPE_CHECKING


if TYPE_CHECKING:
    from typing import List


class TokenType(enum.IntEnum):
    END = 0
    START = 1
    REACTION = 2
    REACTANT = 3



class ForwardData(TypedDict):
    token_types: torch.Tensor
    token_padding_mask: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor

    # condition molecule
    cond_atoms: torch.Tensor
    cond_bonds: torch.Tensor
    cond_atoms_padding_mask: torch.Tensor


class ForwardBatch(TypedDict, total=False):
    token_types: torch.Tensor
    token_padding_mask: torch.Tensor

    reactant_atoms: List[torch.Tensor]
    reactant_bonds: List[torch.Tensor]
    reactant_atoms_padding_mask: List[torch.Tensor]
    
    rxn_indices: torch.Tensor
    # mols_emb: torch.Tensor
    reactant_fps: torch.Tensor

    # condition molecule
    cond_atoms: torch.Tensor
    cond_bonds: torch.Tensor
    cond_atoms_padding_mask: torch.Tensor
    cond_mol_emb: torch.Tensor
