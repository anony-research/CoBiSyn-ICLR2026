from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .mol import Molecule
from .reaction import Reaction

if TYPE_CHECKING:
    from typing import List, Iterable


@dataclass
class RxnNode:
    rxn: Reaction
    product: Molecule
    reactants: Iterable[Molecule]

    source: str = 'retro'
    cost: float = math.inf
    available: bool = False

    def __hash__(self) -> int:
        return hash((self.product, frozenset(self.reactants)))
    
    def __eq__(self, value: object) -> bool:
        assert isinstance(value, RxnNode), f"{value} has type {type(value)} rather than {RxnNode.__name__}"
        return self.product == value.product and frozenset(self.reactants) == frozenset(value.reactants)


@dataclass
class MolNode:
    mol: Molecule
    syn_cost: float = math.inf
    path_cost: float = math.inf
    available: bool = False
    pair: Molecule | None = None
    opt_rxn: RxnNode | None = None
    retro_rxn: RxnNode | None = None    
        
    produced_by: List[RxnNode] = field(default_factory=list)
    consumed_by: List[RxnNode] = field(default_factory=list)



