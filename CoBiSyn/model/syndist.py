from __future__ import annotations
import os, gc
import faiss
import optuna
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import TYPE_CHECKING

from .output_head import mlp
from CoBiSyn.chem.mol import Molecule
from CoBiSyn.data.dataset import DistDataModule

if TYPE_CHECKING:
    from typing import List


class SynDistanceNet(nn.Module):
    def __init__(
        self, 
        dim_in: int,
        dim_emb: int,
        dim_hidden: int,
        num_layers: int = 3,
        max_dist: int = 10,
        drop_rate: float = 0.3
    ):
        super().__init__()
        self._max_dist = max_dist
        self.query_encoder = mlp(
            dim_in=dim_in,
            dim_out=dim_emb,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            dropout_rate=drop_rate
        )
        self.cand_encoder = mlp(
            dim_in=dim_in,
            dim_out=dim_emb,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            dropout_rate=drop_rate
        )

    def forward(self, x, y) -> torch.Tensor:
        query_emb = self.query_encoder(x)
        cand_emb = self.cand_encoder(y)
        return torch.norm(query_emb-cand_emb, p=2, dim=-1)
    
    def triangle_regularizer(self, x, y, z):
        f_xy = self(x, y)
        f_yz = self(y, z)
        f_xz = self(x, z)
        loss = F.relu(f_xz - (f_xy + f_yz))
        return loss.mean()
    
    def get_loss(self, x, y, dist) -> torch.Tensor:
        pred = self(x, y)
        dist_t = torch.log1p(dist)
        pred_t = torch.log1p(pred)
        loss = F.mse_loss(pred_t, dist_t, reduction='mean')
        return loss
    
    @torch.inference_mode()
    def predict(self, mol1: torch.Tensor, mol2: torch.Tensor) -> torch.Tensor:
        return self(mol1, mol2)
    
    @torch.inference_mode()
    def cand_emb(self, fps: torch.Tensor) -> torch.Tensor:
        return self.cand_encoder(fps)
    
    @torch.inference_mode()
    def query_emb(self, fps: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(fps)


class SynDistModel(pl.LightningModule):
    def __init__(self, dim_in, dim_emb, num_layers, dim_hidden, lam_triangle, lam_margin, lr, max_dist=10, drop_rate=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SynDistanceNet(
            dim_in=dim_in,
            dim_emb=dim_emb,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            max_dist=max_dist,
            drop_rate=drop_rate
        )
        self.lam_triangle = lam_triangle
        self.lam_margin = lam_margin
        self.index = None
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, dist = batch
        loss_main = self.model.get_loss(x, y, dist)
        z = y[torch.randperm(y.size(0))]
        loss_triangle = self.model.triangle_regularizer(x, y, z)
        loss_margin = self.model.margin_loss(x, y)

        loss = loss_main + self.lam_triangle * loss_triangle + self.lam_margin * loss_margin
        self.log('train_loss', loss)
        self.log('log-MSE', loss_main)
        self.log('triangle loss', loss_triangle)
        self.log('margin loss', loss_margin)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y, dist = batch
        loss = self.model.get_loss(x, y, dist)

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
    
    @torch.inference_mode()
    def predict(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        return self.model.predict(batch_x, batch_y)

    @torch.inference_mode()
    def cal_candidate_emb(self, fps):
        return self.model.cand_emb(fps)
    
    @torch.inference_mode()
    def cal_query_emb(self, fps):
        return self.model.query_emb(fps)
    
    def load_index(self, path):
        cpu_index = faiss.read_index(path)
        if self.device == 'cpu':
            print(f'Loading f{path} into the cpu...')
            self.index = cpu_index
        else:
            print(f'Loading {path} into the "{self.device}"...')
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.device.index, cpu_index)
        self.index.nprobe = 32


    def query_kNN(self, mol: Molecule, k: int = 5):
        mol_fp = mol.fingerprint.unsqueeze(0)     # (1, 2048)
        mol_emb = self.cal_query_emb(mol_fp)
        dist, indices = self.index.search(mol_emb.cpu(), k)   # (bsz, k, 1)
        return dist, indices
    

    def predict_single_to_multi(self, mol: Molecule, ref_mols: List[Molecule], batch_size=2048) -> torch.Tensor:
        mol_fp = mol.fingerprint.unsqueeze(0)

        dist = torch.zeros(len(ref_mols), device=self.device, dtype=torch.float32)
        for i in range(0, len(ref_mols), batch_size):
            batch_data = ref_mols[i:i+batch_size]
            query_batch = mol_fp.expand(len(batch_data), -1)    # (batch_size, 2048)
            batch = torch.cat(
                [m.fingerprint.unsqueeze(0) for m in batch_data], 
                dim=0
            )
            dist[i:i+len(batch_data)] = self.predict(batch, query_batch).view(-1)
            
        return dist


    def find_nearest(self, mol: Molecule, ref_mols: List[Molecule], batch_size=2048) -> tuple[float, int]:
        mol_fp = mol.fingerprint.unsqueeze(0)

        min_dist = None
        min_idx = None
        for i in range(0, len(ref_mols), batch_size):
            batch_data = ref_mols[i:i+batch_size]
            query_batch = mol_fp.expand(len(batch_data), -1)    # (batch_size, 2048)
            batch = torch.cat(
                [m.fingerprint.view(1, -1) for m in batch_data], 
                dim=0
            )
            dist = self.predict(query_batch, batch).squeeze()
            batch_min_dist, batch_min_idx = torch.min(dist, dim=0)
            if min_dist is None or min_dist > batch_min_dist:
                min_dist = batch_min_dist
                min_idx = i + batch_min_idx

        return min_dist.item(), min_idx.item()


def objective(trial):
    gpu_id = trial.number % 4
    hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024, 2048])
    emb_dim = trial.suggest_categorical('emb_dim', [512, 1024])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    lam_triangle = trial.suggest_float('lam_triangle', 0.01, 0.15)
    lam_margin = trial.suggest_float('lam_margin', 0.01, 0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    data_module = DistDataModule(
        train_path='dataset/rdchiral/dist-train-dataset.json',
        val_path='dataset/rdchiral/dist-validate-dataset.json',
        batch_size=256
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/dist_DE_log_margin/trial_{trial.number}',
        filename='best',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    model = SynDistModel(
        dim_in=2048,
        dim_emb=emb_dim,
        dim_hidden=hidden_dim,
        num_layers=num_layers,
        lam_triangle=lam_triangle,
        lam_margin=lam_margin,
        lr=lr,
        drop_rate=0.3
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu',
        devices=[gpu_id],
        callbacks=[checkpoint, early_stopping],
    )
    trainer.fit(model, data_module)

    val_loss = checkpoint.best_model_score.item()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return val_loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, n_jobs=4)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Val Loss: {trial.value}")
    print("Params:", trial.params)