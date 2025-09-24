from __future__ import annotations
import gc
import json
import logging

from rdkit import RDLogger
import torch
from torch import nn
import torch.nn.functional as F
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import TYPE_CHECKING

from CoBiSyn.chem.mol import Molecule
from CoBiSyn.data.dataset import RetroDataModule

if TYPE_CHECKING:
    from typing import Tuple

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.ERROR)


class Encoder(nn.Module):
    def __init__(self, dim_in=2048, dim_hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )

    def forward(self, x):
        return self.net(x)
    

class Projection(nn.Module):
    def __init__(self, dim_hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 2 * dim_hidden)  # gamma, beta
        )

    def forward(self, target_emb, cond_emb):
        gamma_beta = self.mlp(cond_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * target_emb + beta


class RetroNet(nn.Module):
    def __init__(self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int,
        num_layers: int,
        mode: str = 'pretrain',
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.encoder = Encoder(dim_in=dim_in, dim_hidden=dim_hidden)
        self.proj = Projection(dim_hidden=dim_hidden)
        
        layers = []
        for _ in range(num_layers-1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(dim_hidden, dim_out))
        self.mlp = nn.Sequential(*layers)

        self.mode = mode

    def forward(self, x: torch.Tensor):
        if self.mode == 'pretrain':
            h = self.encoder(x)
        elif self.mode == 'finetune':
            mol, cond = x.chunk(2, dim=-1)
            mol_emb = self.encoder(mol)
            cond_emb = self.encoder(cond)
            h = self.proj(mol_emb, cond_emb)
        else:
            raise NotImplementedError

        return self.mlp(h)

    @torch.inference_mode()
    def predict(
        self, 
        product: torch.Tensor, 
        condition: torch.Tensor, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self(torch.cat([product.unsqueeze(0), condition.unsqueeze(0)], dim=-1))
        probs = F.softmax(output, dim=-1)
        top_k_scores, top_k_indices = torch.topk(probs, top_k)
        return top_k_scores, top_k_indices
    


class RetroModel(pl.LightningModule):
    def __init__(self, output_dim, input_dim, num_layers, hidden_dim, lr, topk, mode='pretrain'):
        super().__init__()
        self.save_hyperparameters()
        self.model = RetroNet(
            dim_in=input_dim,
            dim_out=output_dim,
            num_layers=num_layers,
            dim_hidden=hidden_dim,
            mode=mode
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.mode = mode

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        topk_preds = torch.topk(y_hat, k=self.hparams.topk, dim=-1).indices

        acc_topk = (topk_preds == y.unsqueeze(-1)).any(dim=-1).float().mean()
        loss = self.loss_fn(y_hat.view(-1, y_hat.size(-1)), y.view(-1))


        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'val_acc_top{self.hparams.topk}', acc_topk, prog_bar=True, on_epoch=True, on_step=False)
        return {"val_loss": loss, f"val_acc_top{self.hparams.topk}": acc_topk}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def switch_mode(self, mode):
        self.mode = mode
        self.model.mode = mode
    
    @torch.inference_mode()
    def predict(
        self, 
        product: str | Molecule, 
        condition: str | Molecule, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(product, str):
            product = Molecule(product)
        if isinstance(condition, str):
            condition = Molecule(condition)
        
        top_k_scores, top_k_indices = self.model.predict(product.fingerprint, condition.fingerprint, top_k=top_k)

        return top_k_scores, top_k_indices
    
    @torch.inference_mode()
    def predict_no_cond(
        self, 
        product: str | Molecule, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(product, str):
            product = Molecule(product)
        
        output = self.model(product.fingerprint.unsqueeze(0))
        probs = F.softmax(output, dim=-1)
        top_k_scores, top_k_indices = torch.topk(probs, top_k)
        return top_k_scores, top_k_indices


def objective(trial):
    gpu_id = trial.number % 4
    # hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024, 2048])
    # num_layers = trial.suggest_int("num_layers", 2, 4)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    topk = 10

    data_module = RetroDataModule(
        train_path='dataset/rdchiral/retro-train-dataset.pkl', 
        # train_path='dataset/rdchiral/reactions-train',
        val_path='dataset/rdchiral/retro-validate-dataset.pkl',
        # val_path='dataset/rdchiral/reactions-validate',
        batch_size=1024
    )

    with open('dataset/rdchiral/template2index.json', 'r') as f:
        temp2idx = json.load(f)
    # model = RetroModel(
    #     output_dim=len(temp2idx.items()),
    #     input_dim=2048,
    #     num_layers=num_layers,
    #     hidden_dim=hidden_dim,
    #     lr=lr,
    #     topk=topk,
    #     mode='pretrain'
    # )
    model = RetroModel.load_from_checkpoint(
        'checkpoints/retro_pretrain/trial_0/best.ckpt',
        lr=lr,
        mode='finetune'
    )
    

    early_stop = EarlyStopping(
        monitor=f'val_acc_top{topk}',
        mode='max',
        patience=3,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/retro_{model.mode}/trial_{trial.number}",
        filename="best",  
        monitor=f'val_acc_top{topk}',
        mode='max',
        save_top_k=1,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator='gpu',
        devices=[gpu_id],
        callbacks=[checkpoint, early_stop]
    )
    

    trainer.fit(model, datamodule=data_module)

    val_acc_topk = checkpoint.best_model_score.item()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return val_acc_topk

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=8, n_jobs=4)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Val Acc: {trial.value}")
    print("Params:", trial.params)