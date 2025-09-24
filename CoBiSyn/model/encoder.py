import abc
from typing import TYPE_CHECKING
import torch
from torch import nn

from .transformer.graph_transformer import GraphTransformer
from CoBiSyn.data.common import InferenceBatch

if TYPE_CHECKING:
    from typing import Tuple

class BaseEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    if TYPE_CHECKING:

        def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]: ...


class GraphEncoder(BaseEncoder):
    def __init__(
            self,
            num_atom_classes: int,
            num_bond_classes: int,
            dim: int, 
            depth: int,
            head_dim: int,
            edge_dim: int,
            heads: int,
            rel_pos_emb: bool,
            output_norm: bool
        ):
        super().__init__()
        self._dim = dim
        self._atom_emb = nn.Embedding(num_atom_classes + 1, dim, padding_idx=0)      # TODO: why + 1
        self._bond_emb = nn.Embedding(num_bond_classes + 1, edge_dim, padding_idx=0)
        self._encoder = GraphTransformer(
            dim=dim,
            depth=depth,
            dim_head=head_dim,
            edge_dim=edge_dim,
            heads=heads,
            rel_pos_emb=rel_pos_emb,
            output_norm=output_norm
        )

    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(self, batch: InferenceBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        atoms = self._atom_emb(batch["atoms"])
        bonds = self._bond_emb(batch["bonds"])
        atom_padding_mask = batch["atom_padding_mask"]
        atoms_emb, _ = self._encoder(nodes=atoms, edges=bonds, mask=atom_padding_mask)
        return atoms_emb, atom_padding_mask
