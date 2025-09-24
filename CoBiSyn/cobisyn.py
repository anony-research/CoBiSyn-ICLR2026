from __future__ import annotations
import random
import torch
import time

from typing import TYPE_CHECKING

from CoBiSyn.chem.mol import Molecule
from CoBiSyn.chem.reaction import Reaction
from CoBiSyn.chem.route import SynthesisGraph
from CoBiSyn.data.common import TokenType

if TYPE_CHECKING:
    from typing import List, Set, Tuple, Dict
    from CoBiSyn.model.retro import RetroModel
    from CoBiSyn.model.forward import ForwardModel
    from CoBiSyn.model.syndist import SynDistModel


class CoBiSyn:
    def __init__(
        self,
        bbs_mols: List[Molecule],
        bbs_set: Set[Molecule],
        idx2temp: Dict[int, str],
        retro: RetroModel,
        fwd: ForwardModel,
        syndist: SynDistModel
    ):
        self.bbs_mols = bbs_mols
        self.bbs_set = bbs_set
        self.idx2temp = idx2temp
        self.retro = retro
        self.fwd = fwd
        self.syndist = syndist

        num_reactants = {i: Reaction(temp).num_reactants for i, temp in idx2temp.items()}
        self.num_reactants_mask = torch.zeros((max(num_reactants.values()), len(num_reactants)), dtype=torch.bool, device=fwd.device)
        for i in range(self.num_reactants_mask.shape[0]):
            for j in range(self.num_reactants_mask.shape[1]):
                if num_reactants[j] == i+1:
                    self.num_reactants_mask[i, j] = 1

    
    def run(self, smiles: str, max_round: int = 500):
        start = time.time()
        route = SynthesisGraph(
            target=Molecule(smiles, device=self.retro.device),
            bbs_mols=self.bbs_mols,
            bbs_set=self.bbs_set,
            dist_model=self.syndist
        )

        round = 0
        while not route.finish() and round < max_round and route.retro_frontier:
            # 1. top-down inference
            mol, cond = route.select_optimal_pair()
            reactants, templates = self.topdown_inference(mol, cond)
            route.retro_expansion(
                current=mol,
                precursors=[[Molecule(m, device=self.retro.device) for m in mlist] for mlist in reactants],
                rxns=templates
            )
            
            if route.finish() or len(route.retro_frontier) == 0:
                break

            # 2. bottom-up inference
            cond, mol = route.select_optimal_pair()
            reactants, templates, product = self.bottomup_inference(cond, mol, route.fwd_frontier)
            route.fwd_expansion(
                start=mol,
                reactants=reactants,
                products=[Molecule(p, device=self.fwd.device) for p in product],
                rxn=templates
            )

            round += 1

        end = time.time()
        route.round = round
        route.time = end - start
        return route
        
    def topdown_inference(self, mol: Molecule, cond: Molecule, topk: int = 50) -> Tuple[List[List[str]], List[Reaction]]:
        _, top_k_indices =  self.retro.predict(mol, cond, top_k=topk)
        reactants: List[List[str]] = []
        templates: List[Reaction] = []  
        for i in range(top_k_indices.shape[1]):
            idx = top_k_indices[0][i].item()
            template = Reaction(self.idx2temp[idx], idx)
            retro_reactants = template.retro(mol)
            if len(retro_reactants) > 0:
                for r in retro_reactants:
                    reactants.append(r.split('.'))
                    templates.append(template)
        return reactants, templates

    def bottomup_inference(
        self, 
        cond: Molecule, 
        mol: Molecule,
        bbs: List[Molecule],
        max_len: int = 8
    ) -> Tuple[List[List[Molecule]], List[Reaction], List[str]]:
        cond_smile = cond.smiles
        mol_fp = mol.fingerprint
        token_types, rxn_indices, mol_indices = self.fwd.predict(cond_smile, mol_fp, self.num_reactants_mask, max_len=max_len)

        reactants: List[List[Molecule]] = []
        templates: List[Reaction] = []
        products: List[str] = []
        
        for j in range(rxn_indices.shape[-1]):
            rs: List[Molecule] = [mol]
            for i, token in enumerate(token_types[0]):
                if token == TokenType.REACTANT and i > 1:
                    rs.append(bbs[mol_indices[0][i-1].item()])
                elif token == TokenType.REACTION:
                    idx = rxn_indices[0][i][j].item()
                    rxn: Reaction = Reaction(self.idx2temp[idx], idx)
                    p_list = rxn.fwd(rs)
                    if len(p_list) > 0:
                        p: str = random.choice(p_list)
                        reactants.append(rs)
                        templates.append(rxn)
                        products.append(p)
        return reactants, templates, products