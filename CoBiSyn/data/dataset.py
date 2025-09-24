from __future__ import annotations
import os
import pickle, json
import logging
import numpy as np
from typing import TYPE_CHECKING
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from CoBiSyn.chem.mol import Molecule
from CoBiSyn.data.common import TokenType, ForwardData
from CoBiSyn.data.collate import ForwardCollater

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.ERROR)

if TYPE_CHECKING:
    from typing import List


# ---------- retro model -----------

class RetroDataset(Dataset):
    def __init__(self, fps: np.ndarray, labels: np.ndarray):
        super().__init__()
        self.fps = fps
        self.labels = labels

    def __len__(self):
        return len(self.fps)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.fps[idx]).float(), torch.tensor(self.labels[idx], dtype=torch.long)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        fps, labels = raw
        return cls(fps, labels)
    
    
def transform_retro(data: dict, generator, stage: str):
    if stage == 'pretrain':
        mol_fp = torch.tensor(
            list(generator.GetFingerprint(Chem.MolFromSmiles(data['products']))), 
            dtype=torch.float32
        )
        return (mol_fp, torch.tensor(data['template_id'], dtype=torch.long))
    elif stage == 'finetune':
        mol_fp = torch.tensor(
            list(generator.GetFingerprint(Chem.MolFromSmiles(data['molecule']))), 
            dtype=torch.float32
        )
        cond_fp = torch.tensor(
            list(generator.GetFingerprint(Chem.MolFromSmiles(data['condition']))),
            dtype=torch.float32
        )
        return (torch.cat([mol_fp, cond_fp], dim=0), torch.tensor(data['template'], dtype=torch.long))
    else:
        raise NotImplementedError
    

class RetroDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        print('Loading training dataset...')
        self.train_dataset = RetroDataset.load(train_path)
        print('Loading validation dataset...')
        self.val_dataset = RetroDataset.load(val_path)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=30)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=30)


# --------- forward model ----------

class ForwardDataset(Dataset):
    def __init__(self, reactants: List[np.ndarray], condition: List[str], labels: np.ndarray):
        super().__init__()
        self.reactants = reactants
        self.condition = condition
        self.labels = labels

    def __len__(self):
        return len(self.condition)
    
    def __getitem__(self, idx: int):
        # target molecule
        cond_atoms, cond_bonds = Molecule(self.condition[idx]).featurize()
        # reaction token sequence
        token_types = torch.full((self.reactants[idx].shape[0] + 3,), TokenType.REACTANT, dtype=torch.long)
        token_types[0] = TokenType.START
        token_types[-1] = TokenType.END
        token_types[-2] = TokenType.REACTION
        # reaction template
        rxn_indices = torch.zeros((token_types.shape[0],), dtype=torch.long)
        rxn_indices[-2] = self.labels[idx]
        # reactant fingerprints
        reactant_fps = torch.zeros((self.reactants[idx].shape[0] + 3, 2048), dtype=torch.float32)
        reactant_fps[1:1+self.reactants[idx].shape[0]] = torch.from_numpy(self.reactants[idx]).float()
        return BottomUpData(
            token_types=token_types,
            token_padding_mask=torch.full(
                (token_types.shape[0],), 
                True, 
                dtype=torch.bool
            ),
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            cond_atoms=cond_atoms,
            cond_bonds=cond_bonds,
            cond_atoms_padding_mask=torch.full(
                (cond_atoms.shape[0],), 
                True, 
                dtype=torch.bool
            )
        )

    @classmethod
    def load(cls, path: str):
        with open(path + '-reactants.pkl', 'rb') as f:
            reactants = pickle.load(f)
        with open(path + '-products.pkl', 'rb') as f:
            condition = pickle.load(f)
        return cls(reactants, condition, np.load(path + '-labels-fwd.npy'))


class ForwardDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        print('Loading training dataset...')
        self.train_dataset = ForwardDataset.load(train_path)
        print('Loading validation dataset...')
        self.val_dataset = ForwardDataset.load(val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=30,
            collate_fn=ForwardCollater()
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=30,
            collate_fn=ForwardCollater()
        )


# --------- dist model ---------


class DistDataset(Dataset):
    def __init__(self, dataset: list):
        super().__init__()
        self.dataset = dataset
        self.generator = GetMorganGenerator(radius=2, fpSize=2048)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        raw_data = self.dataset[idx]
        return transform_dist(raw_data, self.generator)
    
    @classmethod
    def from_json(cls, path: os.PathLike):
        with open(path, 'r') as f:
            raw_data = json.load(f)
        return cls(raw_data)

def transform_dist(data, generator):
    mols, dist = data
    mol_x, mol_y = mols
    mol_x_fp = torch.tensor(
        list(generator.GetFingerprint(Chem.MolFromSmiles(mol_x))),
        dtype=torch.float32
    )
    mol_y_fp = torch.tensor(
        list(generator.GetFingerprint(Chem.MolFromSmiles(mol_y))),
        dtype=torch.float32
    )
    return mol_x_fp, mol_y_fp, torch.tensor(dist, dtype=torch.float)


class DistDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        print('Loading training dataset...')
        self.train_dataset = DistDataset.from_json(train_path)
        print('Loading validation dataset...')
        self.val_dataset = DistDataset.from_json(val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=30
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=30
        )