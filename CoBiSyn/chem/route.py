from __future__ import annotations
import math
from typing import TYPE_CHECKING
from collections import deque

from CoBiSyn.chem.mol import Molecule
from CoBiSyn.chem.reaction import Reaction
from CoBiSyn.chem.node import MolNode, RxnNode
from CoBiSyn.model.syndist import SynDistModel

if TYPE_CHECKING:
    from typing import List, Dict, Set, Iterable, Tuple, Sequence
    

class MyDict:
    def __init__(self, bbs_set: Set[Molecule]):
        self.bbs_set = bbs_set
        self.mol_nodes: Dict[Molecule, MolNode] = dict()

    def __contains__(self, key: object) -> bool:
        return key in self.bbs_set or key in self.mol_nodes
    
    def __getitem__(self, key: Molecule):
        if key not in self.mol_nodes:
            if key in self.bbs_set:
                self.mol_nodes[key] = MolNode(key, syn_cost=0, available=True)
            else:
                raise ValueError(f"Incorrect key {key}!")
        return self.mol_nodes[key]
    
    def __setitem__(self, key: Molecule, value: MolNode):
        self.mol_nodes[key] = value


class MyList:
    def __init__(self, bbs_mols: List[Molecule], bbs_set: Set[Molecule]) -> None:
        self.bbs_set = bbs_set
        self.bbs_mols = bbs_mols
        self.bbs_num = len(bbs_mols)
        self.mols_list: List[Molecule] = []
        self.mols_set: Set[Molecule] = set()

    def __getitem__(self, idx: int) -> Molecule:
        if idx < self.bbs_num:
            return self.bbs_mols[idx]
        else:
            return self.mols_list[idx - self.bbs_num]
        
    def __contains__(self, key: Molecule) -> bool:
        return key in self.bbs_set or key in self.mols_set
    
    def __len__(self) -> int:
        return self.bbs_num + len(self.mols_list)
        
    def append(self, mol: Molecule):
        self.mols_list.append(mol)
        self.mols_set.add(mol)



class SynthesisGraph:
    def __init__(
        self,
        target: Molecule,
        bbs_mols: List[Molecule],
        bbs_set: Set[Molecule],
        dist_model: SynDistModel,
    ):
        self.mol_nodes = MyDict(bbs_set)
        self.rxn_nodes: Set[RxnNode] = set()

        self.retro_frontier: Set[Molecule] = set()
        self.fwd_frontier: MyList = MyList(bbs_mols, bbs_set)

        self.dist_model = dist_model
        self.set_target(target)

        self.round: int = 0
        self.time: float = -1

    
    def _create_mol_node(self, m: Molecule, produced_by: RxnNode | None = None) -> MolNode:
        m_node = MolNode(mol=m)
        if produced_by is not None:
            m_node.produced_by.append(produced_by)
            m_node.opt_rxn = produced_by
            m_node.syn_cost = produced_by.cost
        self.mol_nodes[m] = m_node
        return m_node

    
    # ---------- property operations ----------
    def _find_pair_retro(self, m: Molecule) -> None:
        dist, indices = self.dist_model.query_kNN(m)
        if len(self.fwd_frontier.mols_list):
            dist_, idx_ = self.dist_model.find_nearest(m, self.fwd_frontier.mols_list)
        else:
            dist_, idx_ = math.inf, math.inf
        idx_ += self.fwd_frontier.bbs_num

        if dist[0][0] <= dist_:
            min_dist = dist[0][0]
            min_idx = indices[0][0]
        else:
            min_dist = dist_
            min_idx = idx_

        pair: Molecule = self.fwd_frontier[min_idx]

        m_node = self.mol_nodes[m]
        m_node.syn_cost = min_dist + self.mol_nodes[pair].syn_cost
        m_node.pair = pair

    def _update_pair_fwd(self, m: Molecule):
        retro_frontier_list: List[Molecule] = list(self.retro_frontier)
        dist = self.dist_model.predict_single_to_multi(mol=m, ref_mols=retro_frontier_list).cpu().numpy()

        m_node = self.mol_nodes[m]
        changed: List[Molecule] = []
        for i, mol in enumerate(retro_frontier_list):
            mol_node = self.mol_nodes[mol]
            if dist[i] + m_node.syn_cost < mol_node.syn_cost:
                mol_node.pair = m
                mol_node.syn_cost = dist[i] + m_node.syn_cost
                changed.append(mol)
        
        self._update_syn_cost(changed)
        

    def _cal_reaction_cost(self, rxn: RxnNode) -> None:
        total = 1.0
        for m in rxn.reactants:
            total += self.mol_nodes[m].syn_cost
        rxn.cost = total

    def _get_min_cost_reaction(self, m: Molecule) -> RxnNode:
        min_cost_rxn = min(self.mol_nodes[m].produced_by, key=lambda x: x.cost)
        return min_cost_rxn
    
    def _update_syn_cost(self, currents: Iterable[Molecule]) -> None:
        """
        Propagate the influence of syn_cost change.
        """
        # 1. propagate the syn_cost update to the roots
        for current in currents:
            current_node = self.mol_nodes[current]

            visited_m: Set[Molecule] = set([current])
            queue: deque[RxnNode] = deque([r for r in current_node.consumed_by])
            while queue:
                r = queue.popleft()
                self._cal_reaction_cost(r)
                p_node = self.mol_nodes[r.product]
                visited_m.add(r.product)
                min_cost_rxn = self._get_min_cost_reaction(r.product)
                if r != p_node.opt_rxn and r != min_cost_rxn:
                    continue
                p_node.opt_rxn = min_cost_rxn
                p_node.syn_cost = min_cost_rxn.cost
                for r in p_node.consumed_by:
                    if r.product not in visited_m:
                        queue.append(r)

        # 2. propagate the path_cost update to the leaves
        self._update_path_cost()


    def _update_path_cost(self):
        target_node = self.mol_nodes[self.target]
        target_node.path_cost = target_node.syn_cost
        visited_m: set[Molecule] = set([self.target])
        queue_syn: deque[Tuple[RxnNode, float]] = deque([(r, r.cost) for r in target_node.produced_by])
        while queue_syn:
            r, c = queue_syn.popleft()
            for m in r.reactants:
                m_node = self.mol_nodes[m]
                if m in visited_m and c >= m_node.path_cost:
                    continue                
                m_node.path_cost = c
                visited_m.add(m)
                for r2 in m_node.produced_by:
                    queue_syn.append((r2, c - m_node.syn_cost + r2.cost))


    def update_cost_and_available(self, current: Molecule):
        current_node = self.mol_nodes[current]

        # 1. update available property
        for r in current_node.produced_by:
            for m in r.reactants:
                # print(f'{self.mol_nodes[m].mol.smiles}: {m in self.fwd_frontier}')
                if m in self.fwd_frontier:
                    self.mol_nodes[m].available = True
            r.available = all([self.mol_nodes[m].available for m in r.reactants])
        current_node.available = any([r.available for r in current_node.produced_by])
        if current_node.available:
            self._update_available(current)
        if self.finish():
            return

        # 2. initialize the syn_cost and pair for newly expanded molecules
        for r in current_node.produced_by:
            for m in r.reactants:
                m_node = self.mol_nodes[m]
                if m_node.syn_cost == math.inf:
                    self._find_pair_retro(m)
            self._cal_reaction_cost(r)

        # 3. update the syn_cost for current
        min_cost_rxn = self._get_min_cost_reaction(current)
        current_node.opt_rxn = min_cost_rxn
        current_node.path_cost += (min_cost_rxn.cost - current_node.syn_cost)
        current_node.syn_cost = min_cost_rxn.cost

        # 4. propagate the influenc brought by syn_cost change of current node
        self._update_syn_cost([current])
        

    def _update_available(self, current: Molecule):
        """
        Update the change of available property induced by current molecule being available.
        """
        queue: deque[Molecule] = deque([current])
        while queue:
            m = queue.popleft()
            m_node = self.mol_nodes[m]
            for r in m_node.consumed_by:
                if r.available:
                    continue
                r.available = all([self.mol_nodes[m].available for m in r.reactants])
                if r.available:
                    p_node = self.mol_nodes[r.product]
                    if not p_node.available:
                        p_node.available = True
                        queue.append(r.product)


    def _reachable(self, start: Molecule, end: Molecule) -> bool:
        if start not in self.mol_nodes or end not in self.mol_nodes:
            return False
        if start == end:
            return True
        visited_m: Set[Molecule] = set([start])
        queue: deque[RxnNode] = deque(self.mol_nodes[start].produced_by)
        while queue:
            r = queue.popleft()
            for m in r.reactants:
                if m == end:
                    return True
                if m not in visited_m:
                    visited_m.add(m)
                    for r2 in self.mol_nodes[m].produced_by:
                        queue.append(r2)
        return False
    

    # ---------- frontier operations ----------

    def set_target(self, target: Molecule) -> None:
        self.target: Molecule = target
        if target not in self.mol_nodes:
            self._create_mol_node(target)
            self._find_pair_retro(target)
        self.mol_nodes[target].path_cost = self.mol_nodes[target].syn_cost
        self.retro_frontier = {target}


    def retro_expansion(self, current: Molecule, precursors: Sequence[Iterable[Molecule]], rxns: Iterable[Reaction]):
        # structure update
        self.retro_frontier.remove(current)
        non_null = False
        for i, rxn in enumerate(rxns):
            if any(m == self.target for m in precursors[i]):
                continue
            if any(self._reachable(m, current) for m in precursors[i]):
                continue
            if any(m._rdmol is None for m in precursors[i]):
                continue
            r = RxnNode(rxn=rxn, reactants=precursors[i], product=current)
            if r in self.rxn_nodes:
                continue
            self.rxn_nodes.add(r)
            non_null = True

            self.mol_nodes[current].produced_by.append(r)
            for m in precursors[i]:
                if m not in self.mol_nodes:
                    m_node = self._create_mol_node(m)
                    self.retro_frontier.add(m)
                else:
                    m_node = self.mol_nodes[m]
                m_node.consumed_by.append(r)

        # property update
        if non_null:
            self.update_cost_and_available(current)
        

    def fwd_expansion(self, start: Molecule, reactants: List[List[Molecule]], products: List[Molecule], rxn: List[Reaction]):
        for i, rset in enumerate(reactants):
            if products[i]._rdmol is None:
                continue
            r = RxnNode(rxn=rxn[i], reactants=reactants[i] + [start], product=products[i], source='fwd')
            if r in self.rxn_nodes:
                continue
            self.rxn_nodes.add(r)
            self._cal_reaction_cost(r)

            product = products[i]
            if product not in self.mol_nodes:
                p_node = self._create_mol_node(product, produced_by=r)
                self.fwd_frontier.append(product)
                self._update_pair_fwd(product)

            else:
                p_node = self.mol_nodes[product]
                p_node.produced_by.append(r)
                min_cost_rxn = self._get_min_cost_reaction(product)

                if min_cost_rxn != r:
                    return
                
                changed: List[Molecule] = []
                changed.append(product)
                for m in self.retro_frontier:
                    m_node = self.mol_nodes[m]
                    if m_node.pair == product:
                        m_node.syn_cost += (r.cost - p_node.syn_cost)
                        changed.append(m)

                p_node.opt_rxn = min_cost_rxn
                p_node.syn_cost = min_cost_rxn.cost

                self._update_syn_cost(changed)


    # ---------- common api ----------
    def select_optimal_pair(self) -> Tuple[Molecule, Molecule]:
        mol = min(self.retro_frontier, key=lambda m: self.mol_nodes[m].path_cost)
        return mol, self.mol_nodes[mol].pair
    
    def finish(self) -> bool:
        return self.mol_nodes[self.target].available
    
    def success_route(self) -> List[Tuple[str, str]]:
        route = []
        queue: deque[Molecule] = deque([self.target])
        visited: Set[Molecule] = set()
        while queue:
            m = queue.popleft()
            if m in visited:
                continue

            visited.add(m)
            m_node = self.mol_nodes[m]
            if all(rxn.available == False for rxn in m_node.produced_by):
                continue
            
            rxn = min([r for r in m_node.produced_by if r.available], key=lambda x: x.cost)    
            route.append((m_node.mol.smiles + '>>' + '.'.join([react.smiles for react in rxn.reactants]), rxn.rxn.idx))
            for react in rxn.reactants:
                queue.append(react)
        return route