from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from functools import cached_property
from itertools import permutations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mol import Molecule
    from typing import List


class Reaction:
    def __init__(self, smarts: str, idx: int | None = None):
        self.smarts = smarts
        self.idx = idx

    @property
    def _fwd_smarts(self) -> str:
        return self.smarts.split('>>')[1] + '>>' + self.smarts.split('>>')[0]

    @cached_property
    def _retro_reaction(self) -> rdChemReactions.ChemicalReaction:
        rxn = rdChemReactions.ReactionFromSmarts(self.smarts)
        return rxn

    @cached_property
    def _fwd_reaction(self) -> rdChemReactions.ChemicalReaction:
        rxn = rdChemReactions.ReactionFromSmarts(self._fwd_smarts)
        return rxn

    @property
    def num_reactants(self) -> int:
        return len(self.smarts.split('>>')[1].split('.'))
    
    def retro(self, mol: Molecule) -> List[str]:
        try:
            reactants = self._retro_reaction.RunReactants([mol._rdmol])
            unique_reactants = set()
            for rset in reactants:
                smiles = [Chem.MolToSmiles(r, canonical=True, isomericSmiles=True) for r in rset]
                unique_reactants.add(".".join(sorted(smiles)))
            return list(unique_reactants)
        except Exception as e:
            return []
        
    
    def fwd(self, mols: List[Molecule]) -> List[str]:
        reactants = [mol._rdmol for mol in mols]
        unique_products = set()
        for item in list(permutations(reactants)):
            products = self._fwd_reaction.RunReactants(item)
            for pset in products:
                smiles = [Chem.MolToSmiles(p, canonical=True, isomericSmiles=True) for p in pset]
                unique_products.add(".".join(sorted(smiles)))
        return list(unique_products)
    