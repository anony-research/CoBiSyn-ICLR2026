import os
from rdkit import Chem
from functools import cached_property
from collections.abc import Sequence
import torch
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm.auto import tqdm

from .feature import atom_feature, bond_feature


class Molecule:
    def __init__(self, smiles: str | None = None, index: int = None, device: str | torch.device = 'cpu'):
        self._smiles : str | None = smiles
        self._index: int = index
        self._device = torch.device(device) if isinstance(device, str) else device
    
    @property
    def smiles(self) -> str:
        return self._smiles
    
    @property
    def index(self) -> int:
        return self._index
    
    @property
    def device(self) -> torch.device:
        """
        The device on which the fingerprint tensor is stored.
        If the device is not set, it defaults to 'cpu'.
        """
        return self._device
    
    def set_index(self, index: int):
        self._index = index

    def set_device(self, device: str | torch.device):
        self._device = torch.device(device) if isinstance(device, str) else device

    @cached_property
    def _rdmol(self) -> Chem.Mol:
        return Chem.MolFromSmiles(self._smiles) if self._smiles is not None else None
    
    @cached_property
    def _rdmol_no_hs(self) -> Chem.Mol:
        return Chem.RemoveHs(self._rdmol)
    
    @cached_property
    def _crdmol(self) -> Chem.Mol:
        return Chem.MolFromSmiles(self.csmiles) if self._smiles is not None else None
    
    @cached_property
    def is_valid(self) -> bool:
        return self._rdmol is not None
    
    @cached_property
    def num_atom(self) -> int:
        return self._rdmol.GetNumAtoms()
    
    @cached_property
    def csmiles(self) -> str:
        return Chem.MolToSmiles(self._rdmol, canonical=True, isomericSmiles=False)
    
    @cached_property
    def major_molecule(self) -> "Molecule":
        if "." in self.smiles:
            segs = self.smiles.split(".")
            segs.sort(key=lambda a: -len(a))
            return Molecule(segs[0], index=self.index)
        return self
    
    @cached_property
    def fingerprint(self) -> torch.Tensor:
        """
        The Morgan fingerprint of the molecule, shaped as a 1D tensor of size 2048.
        """
        generator = GetMorganGenerator(radius=2, fpSize=2048)
        return torch.tensor(
            list(generator.GetFingerprint(self._rdmol)),
            dtype=torch.float32,
            device=self.device
        )
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Molecule) and self.smiles == other.smiles
    
    def __str__(self) -> str:
        return f"Molecule: {self.csmiles}"
    
    def __hash__(self) -> int:
        return hash(self.smiles)
        
    def featurize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the molecule into feature vectors of atoms and bonds. Atoms are 
        represented as a 1D tensor of atomic numbers, and bonds are represented
        as a 2D tensor where each entry (i, j) represents the bond type between
        atom i and atom j.

        Returns:
            `atoms_t`: A tensor containing atom features.
            `bonds_t`: A tensor containing bond features.
        """
        mol = self._rdmol_no_hs
        atoms = mol.GetAtoms()

        atoms_t = torch.zeros([len(atoms)], dtype=torch.long, device=self.device)
        bonds_t = torch.zeros([len(atoms), len(atoms)], dtype=torch.long, device=self.device)

        for atom in atoms:
            idx = atom.GetIdx()
            atoms_t[idx] = atom_feature(atom)
            for atom_j in atoms:
                jdx = atom_j.GetIdx()
                bond = mol.GetBondBetweenAtoms(idx, jdx)
                if bond is not None:
                    bonds_t[idx, jdx] = bond_feature(bond)
        
        return atoms_t, bonds_t
    
    @classmethod
    def from_rdmol(cls, mol: Chem.Mol) -> "Molecule":
        return cls(Chem.MolToSmiles(mol))


class MolSequence(Sequence):
    def __init__(self, mols: Sequence[Molecule] = None):
        self._mols = list(mols) if mols != None else []

    def __iter__(self):
        return iter(self._mols)
    
    def __len__(self):
        return len(self._mols)
    
    def __getitem__(self, index):
        return self._mols[index]
    
    def append(self, mols: Molecule | Sequence[Molecule] | None):
        if isinstance(mols, Molecule):
            self._mols.append(mols)
        elif isinstance(mols, Sequence) and all(isinstance(i, Molecule) for i in mols):
            self._mols.extend(list(mols))
        elif mols is None:
            self._mols.append(Molecule(None))
        else:
            raise TypeError("MolSequence.append() excepts an argument of type Molecule or Sequence[Molecule]")

    @classmethod
    def from_sdf(cls, file_path: os.PathLike, drop_duplicates: bool = True):
        """
        Read molecules from an SDF file and return a MolSequence object.
        """
        supplier = Chem.SDMolSupplier(file_path)
        mols = []
        cnt = 0
        visited : set[str] = set()
        for rdmol in tqdm(supplier, desc="Reading Molecules"):
            if rdmol is not None:
                mol = Molecule.from_rdmol(rdmol)
                mol.set_index(cnt)
                mol = mol.major_molecule
                if drop_duplicates and mol.csmiles in visited:
                    continue
                mols.append(mol)
                visited.add(mol.csmiles)
                cnt += 1
        return cls(mols)
    
    def fingerprints(self) -> torch.Tensor:
        fps = torch.empty((0, 2048))
        for mol in tqdm(self, desc='calculating fingerprints'):
            fps = torch.cat([fps, mol.fingerprint.unsequeeze(0)], dim=0)
        return fps