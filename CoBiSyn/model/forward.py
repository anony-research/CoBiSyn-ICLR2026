from __future__ import annotations
import os
import h5py
import faiss
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import TYPE_CHECKING

from .transformer.graph_transformer import GraphTransformer
from .transformer.positional_encoding import PositionalEncoding
from .output_head import ClassifierHead, MoleculeHead, mlp
from CoBiSyn.data.common import TokenType, ForwardBatch
from CoBiSyn.data.dataset import ForwardDataModule
from CoBiSyn.chem.mol import Molecule

if TYPE_CHECKING:
    from typing import List, Tuple, Dict


class PackedFPs:
    def __init__(self, fps: np.ndarray, device: torch.device):
        self.fps = fps
        self.device = device

    def __len__(self): return len(self.fps)

    def __getitem__(self, idx):
        packed = self.fps[idx]
        unpacked = np.unpackbits(packed, axis=-1).astype(np.float32)
        return torch.tensor(unpacked, device=self.device, dtype=torch.float32)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_atom_classes: int,
        num_bond_classes: int,
        dim: int,
        depth: int,
        dim_head: int,
        edge_dim: int,
        heads: int,
        rel_pos_emb: bool,
        output_norm: bool,
    ):
        super().__init__()
        self._dim = dim
        self.atom_emb = nn.Embedding(num_atom_classes + 1, dim, padding_idx=0)
        self.bond_emb = nn.Embedding(num_bond_classes + 1, edge_dim, padding_idx=0)
        self.enc = GraphTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            edge_dim=edge_dim,
            heads=heads,
            rel_pos_emb=rel_pos_emb,
            output_norm=output_norm,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        if "cond_atoms" not in batch or "cond_bonds" not in batch or "cond_atoms_padding_mask" not in batch:
            raise ValueError("atoms, bonds and atoms_padding_mask must be in batch")
        atoms = batch["cond_atoms"]
        bonds = batch["cond_bonds"]
        atom_padding_mask = batch["cond_atoms_padding_mask"]

        atom_emb = self.atom_emb(atoms)
        bond_emb = self.bond_emb(bonds)
        node, _ = self.enc(nodes=atom_emb, edges=bond_emb, mask=atom_padding_mask)
        return node, atom_padding_mask


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        num_templates: int,
        dim_fp_embed_hidden: int = 512,
        output_norm: bool = False,
        pe_max_len: int = 16
    ):
        super().__init__()
        self._d_model = d_model
        self._in_token = nn.Embedding(max(TokenType) + 1, d_model)
        self._in_reaction = nn.Embedding(num_templates, d_model)
        self._in_reactant = mlp(
            dim_in=2048,
            dim_out=d_model,
            dim_hidden=dim_fp_embed_hidden,
            num_layers=3
        )
        self._pe = PositionalEncoding(d_model=d_model, max_len=pe_max_len)
        self._decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if output_norm else None
        )

    def embed(
        self, 
        token_types: torch.Tensor,
        emb_reactants: torch.Tensor,
        rxn_indices: torch.Tensor,
        strategy: str = 'substitute'
    ) -> torch.Tensor:
        """
        Calculate the embedding for the input tokens (reaction sequence).
        Args:
            token_types (torch.Tensor): The token types, which can be either reactants or reactions.
            emb_reactants (torch.Tensor): The embedding for the reactants.
            rxn_indices (torch.Tensor): The indices of the reactions.
        Returns:
            torch.Tensor: The embedded tokens.
        """
        emb_token = self._in_token(token_types)
        emb_rxn = self._in_reaction(rxn_indices[:, :, 0])
        token_types_expand = token_types.unsqueeze(-1).expand([token_types.size(0), token_types.size(1), self._d_model])
        if strategy == 'substitute':
            emb_token = torch.where(token_types_expand == TokenType.REACTANT, emb_reactants, emb_token)
            emb_token = torch.where(token_types_expand == TokenType.REACTION, emb_rxn, emb_token)
        elif strategy == 'add':
            content_emb = torch.zeros_like(emb_token)
            content_emb = torch.where(token_types_expand == TokenType.REACTANT, emb_reactants, content_emb)
            content_emb = torch.where(token_types_expand == TokenType.REACTION, emb_rxn, content_emb)
            emb_token += content_emb
        emb_token = self._pe(emb_token)
        return emb_token

    def forward(
        self, 
        condition_mol_emb: torch.Tensor,
        condition_padding_mask: torch.Tensor,
        token_types: torch.Tensor,
        reactant_fps: torch.Tensor,
        rxn_indices: torch.Tensor,
        token_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, seqlen = token_types.size()
        mols_emb = self._in_reactant(reactant_fps)  # fingerprint embedding
        x = self.embed(token_types, mols_emb, rxn_indices)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=x.size(1),
            dtype=x.dtype,
            device=x.device
        )
        tgt_key_padding_mask = (
            torch.zeros([bsz, seqlen], dtype=causal_mask.dtype, device=causal_mask.device)
            .masked_fill_(token_padding_mask, -torch.finfo(causal_mask.dtype).max)
            if token_padding_mask is not None else None
        )
        tgt_key_padding_mask = token_padding_mask 
        y = self._decoder(
            tgt=x,
            memory=condition_mol_emb,  # output of encoder
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=condition_padding_mask
        )
        return y
    


class BottomUpNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        num_templates: int,
        rxn_dim_hidden: int,
        token_dim_hidden: int,
        fp_dim: int = 2048,
        output_norm: bool = False
    ):
        super().__init__()
        self._encoder = GraphEncoder(
            num_atom_classes=100,
            num_bond_classes=10,
            dim=d_model,
            depth=num_layers,
            dim_head=64,
            edge_dim=128,
            heads=nhead,
            rel_pos_emb=False,
            output_norm=False
        )
        self._decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_templates=num_templates,
            output_norm=output_norm
        )
        self._token_head = ClassifierHead(
            dim_in=d_model,
            num_class=len(TokenType),
            dim_hidden=token_dim_hidden
        )
        self._rxn_head = ClassifierHead(
            dim_in=d_model,
            num_class=num_templates,
            dim_hidden=rxn_dim_hidden
        )
        self._mol_head = MoleculeHead(
            dim_in=d_model,
            dim_out=fp_dim
        )
        self.index: faiss.IndexIVFPQ | None = None

    def encode(self, batch: ForwardBatch):
        return self._encoder(batch)
    
    def init_faiss_index(self, faiss_path: os.PathLike, gpu_id: int, nprobe: int = 32):
        index = faiss.read_index(faiss_path)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        self.index.nprobe = nprobe
        
    @torch.inference_mode()
    def predict_next_token(
        self,
        condition_mol_emb: torch.Tensor | None,
        condition_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        reference_fps: PackedFPs,
        token_padding_mask: torch.Tensor | None = None,
    ):
        h = self._decoder(
            condition_mol_emb=condition_mol_emb,
            condition_padding_mask=condition_padding_mask,
            token_types=token_types,
            reactant_fps=reactant_fps,
            rxn_indices=rxn_indices,
            token_padding_mask=token_padding_mask
        )
        h_next = h[:, -1]   # (bsz, h_dim)

        token_logits = self._token_head.predict(h_next)
        reaction_logits = self._rxn_head.predict(h_next)
        _, idx = self._mol_head.retrieve(h_next, reference_fps, faiss_index=self.index)
        return token_logits, reaction_logits, idx
    
    @torch.inference_mode()
    def predict(
        self,
        condition_mol_emb: torch.Tensor,
        condition_padding_mask: torch.Tensor,
        mol_fp: torch.Tensor | None,
        reference_fps: PackedFPs,
        num_reactants_mask: torch.Tensor,
        max_len: int = 8,
        rxn_topk: int = 50,
    ):
        bsz = condition_mol_emb.size(0)
        if mol_fp is None:
            token_types = torch.full([bsz, 1], fill_value=TokenType.START, dtype=torch.long, device=condition_mol_emb.device)
            reactant_fps = torch.full([bsz, 1, 2048], fill_value=0, dtype=torch.float32, device=condition_mol_emb.device)
            rxn_indices = torch.full([bsz, 1, rxn_topk], fill_value=0, dtype=torch.long, device=condition_mol_emb.device)
        else:
            token_types = torch.full([bsz, 2], fill_value=TokenType.REACTANT, dtype=torch.long, device=mol_fp.device)
            token_types[:, 0] = TokenType.START
            reactant_fps = torch.full([bsz, 2, 2048], fill_value=0, dtype=torch.float32, device=mol_fp.device)
            reactant_fps[:, 1, :] = mol_fp
            rxn_indices = torch.full([bsz, 2, rxn_topk], fill_value=0, dtype=torch.long, device=mol_fp.device)
        mol_idices = torch.empty((bsz, 0), dtype=torch.long, device=condition_mol_emb.device)

        finish = torch.zeros(bsz, dtype=torch.bool, device=condition_mol_emb.device)
        while token_types.size(1) < max_len:
            token_logits, reaction_logits, idx = self.predict_next_token(
                condition_mol_emb=condition_mol_emb,
                condition_padding_mask=condition_padding_mask,
                token_types=token_types,
                rxn_indices=rxn_indices,
                reactant_fps=reactant_fps,
                reference_fps=reference_fps
            )
            min_idx = idx[:, 0:1]     # (bsz, 1)
            retrieved_reactants = reference_fps[min_idx] 
            mol_idices = torch.cat([mol_idices, min_idx], dim=-1)

            token_logits[:, 0] = float('-inf')
            token_types = torch.cat([token_types, token_logits.argmax(dim=-1, keepdim=True)], dim=1)

            reaction_logits_masked = reaction_logits.masked_fill(~num_reactants_mask[token_types.size(1)-3, :], float('-inf'))
            _, rxn_topk_indices = torch.topk(reaction_logits_masked, k=rxn_topk, dim=-1)
            rxn_indices = torch.cat([rxn_indices, rxn_topk_indices.view(-1, 1, rxn_topk)], dim=1)

            reactant_fps = torch.cat([reactant_fps, retrieved_reactants.view(-1, 1, 2048)], dim=1)

            finish |= token_types[:, -1] == TokenType.REACTION
            if finish.all():
                break 
        
        return token_types, rxn_indices, mol_idices
    

    def get_loss(self, batch: ForwardBatch) -> Dict[str, torch.Tensor]:
        token_types = batch['token_types']
        token_padding_mask = batch['token_padding_mask']
        reactant_fps = batch['reactant_fps']
        rxn_indices = batch['rxn_indices']
        condition_mol_emb, condition_padding_mask = self.encode(batch)

        h = self._decoder(
            condition_mol_emb=condition_mol_emb,
            condition_padding_mask=condition_padding_mask,
            token_types=token_types,
            token_padding_mask=token_padding_mask,
            reactant_fps=reactant_fps,
            rxn_indices=rxn_indices
        )[:, :-1]

        token_types_gt = token_types[:, 1:]
        rxn_indices_gt = rxn_indices[:, 1:]
        reactant_fps_gt = reactant_fps[:, 1:]
        
        loss_dict: dict[str, torch.Tensor] = {}
        loss_dict['token'] = self._token_head.get_loss(h, token_types_gt, None)
        loss_dict['rxn'] = self._rxn_head.get_loss(h, rxn_indices_gt, token_types_gt == TokenType.REACTION)
        loss_dict['mol'] = self._mol_head.get_loss(h, reactant_fps_gt, token_types_gt == TokenType.REACTANT)

        return loss_dict
    

class ForwardModel(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int, 
        dim_feedforward: int,
        num_layers: int,
        num_templates: int,
        rxn_dim_hidden: int = 512,
        token_dim_hidden: int = 512,
        ref_fps: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = BottomUpNet(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_templates=num_templates,
            rxn_dim_hidden=rxn_dim_hidden,
            token_dim_hidden=token_dim_hidden
        )
        if ref_fps is not None:
            self._ref_fps = ref_fps
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.model.get_loss(batch)
        token_loss = loss_dict['token']
        rxn_loss = loss_dict['rxn']
        mol_loss = loss_dict['mol']
        loss = token_loss + rxn_loss + mol_loss
        self.log('token_loss', token_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('rxn_loss', rxn_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('mol_loss', mol_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss_dict = self.model.get_loss(batch)
        loss = sum(loss_dict.values())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def load_ref_fps(self, fps_path: os.PathLike, faiss_path: os.PathLike | None = None):
        with h5py.File(fps_path, 'r') as f:
            self.ref_fps = PackedFPs(f['bbs_fps'][:], self.device)
        if faiss_path is not None:
            self.init_faiss(faiss_path)

    def init_faiss(self, faiss_path: os.PathLike):
        self.model.init_faiss_index(
            faiss_path=faiss_path,
            gpu_id=self.device.index
        )

    @torch.inference_mode()
    def predict(self, cond: str, mol: str | torch.Tensor | None, num_reactants_mask: torch.Tensor, max_len: int = 8):
        cond_atoms, cond_bonds = Molecule(cond, device=self.device).featurize()
        cond_dict = {
            'cond_atoms': cond_atoms.unsqueeze(0),
            'cond_bonds': cond_bonds.unsqueeze(0),
            'cond_atoms_padding_mask': torch.full((cond_atoms.shape[0],), True, dtype=torch.bool, device=self.device).unsqueeze(0)
        }
        cond_mol_emb, cond_padding_mask = self.model.encode(cond_dict)
        if isinstance(mol, str):
            mol = Molecule(mol, device=self.device).fingerprint.unsqueeze(0)
        
        return self.model.predict(
            condition_mol_emb=cond_mol_emb,
            condition_padding_mask=cond_padding_mask,
            mol_fp=mol,
            reference_fps=self.ref_fps,
            num_reactants_mask=num_reactants_mask,
            max_len=max_len
        )
    
def objective(trial):
    gpu_id = trial.number % 4

    num_layers = trial.suggest_int("num_layers", 2, 6)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    data_module = ForwardDataModule(
        train_path='dataset/rdchiral/fwd-train-dataset.json',
        val_path='dataset/rdchiral/fwd-validate-dataset.json',
        batch_size=128
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/fwd_ema_2/trial_{trial.number}',
        filename='best',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    with open('dataset/rdchiral/template2index.json', 'r') as f:
        temp2idx = json.load(f)
    model = BottomUpModel(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_layers=num_layers,
        num_templates=len(temp2idx.items()),  
        rxn_dim_hidden=512,
        token_dim_hidden=512,
        lr=lr
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
    study.optimize(objective, n_trials=15, n_jobs=4)
   

    print("Best trial:")
    trial = study.best_trial
    print(f"  Val Loss: {trial.value}")
    print("Params:", trial.params)