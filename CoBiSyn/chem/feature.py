from rdkit import Chem
from collections import defaultdict


BOND_TYPE_MAPPING = defaultdict(lambda: 5, {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4
})


def atom_feature(atom: Chem.rdchem.Atom):
    if atom is None:
        return 0
    return atom.GetAtomicNum()


def bond_feature(bond: Chem.rdchem.Bond):
    if bond is None:
        return 0
    return BOND_TYPE_MAPPING[bond.GetBondType()]
