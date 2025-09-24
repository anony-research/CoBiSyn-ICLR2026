from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import faiss
from typing import TYPE_CHECKING

from CoBiSyn.chem.utils import tanimoto_one_to_many

if TYPE_CHECKING:
    from typing import Tuple


def mlp(
    dim_in: int,
    dim_out: int,
    dim_hidden: int,
    num_layers: int = 3,
    dropout_rate: float = 0.3
) -> nn.Sequential:
    layers = [
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU()
    ]
    for _ in range(num_layers-1):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)


class ClassifierHead(nn.Module):
    def __init__(self, dim_in: int, num_class: int, dim_hidden: int | None = None):
        super().__init__()
        dim_hidden = dim_hidden or 2 * dim_in
        self._mlp = mlp(dim_in, num_class, dim_hidden)

    @torch.inference_mode()
    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """
        Return logits.
        """
        return self._mlp(h)
    
    def get_loss(self, h: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self._mlp(h)
        logits_flat = logits.reshape(-1, logits.size(-1))
        gt_flat = gt.reshape(-1)
        if mask is not None:
            mask_flat = mask.reshape(-1)
            total = mask_flat.sum().type_as(logits_flat) + 1e-6
            loss = (F.cross_entropy(logits_flat, gt_flat, reduction='none') * mask_flat).sum() / total
        else:
            loss = F.cross_entropy(logits_flat, gt_flat)
        return loss
    

class MoleculeHead(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int | None = None):
        super().__init__()
        dim_hidden = dim_hidden or 2 * dim_in
        self._mlp = mlp(dim_in, dim_out, dim_hidden)
    
    def get_loss(
        self, 
        h: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ):
        bsz, seqlen, _ = h.shape
        fps_logits = self._mlp(h)    # (bsz, seqlen, fp_dim)
        loss = F.binary_cross_entropy_with_logits(
            fps_logits,
            target,
            reduction='none'
        ).sum(dim=-1)  # (bsz, seqlen)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
    
    def predict(self, h: torch.Tensor) -> torch.Tensor:
        y_fp = torch.sigmoid(self._mlp(h))
        return y_fp
    
    @torch.inference_mode()
    def retrieve(
        self,
        h: torch.Tensor,
        mol_fps: torch.Tensor,
        faiss_index: faiss.Index | None = None,
        topk: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h (torch.Tensor): The input tensor.
            mol_fps (torch.Tensor): The reference fingerprints.
            topk (int): The number of top results to retrieve.
        Returns:
            tuple: A tuple containing the distances and the indices of top results.
        """
        fp = self.predict(h)    # (bsz, fp_dim)
        if faiss_index is None:
            pwdist = torch.cdist(mol_fps, fp, p=1)
            dist, indices = torch.topk(pwdist, topk, dim=0, largest=False, sorted=True)
            return dist, indices
        else:
            fp_np = fp.cpu().numpy()
            _, indices = faiss_index.search(fp_np, 10*topk)
            sims = tanimoto_one_to_many(fp.view(-1), mol_fps[indices[0], :])
            vals, idxs = torch.topk(sims, topk)
            indices_t = torch.tensor(indices, dtype=torch.long, device=h.device)
            return vals, indices_t[:, idxs]
        