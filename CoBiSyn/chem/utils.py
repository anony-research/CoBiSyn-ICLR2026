import numpy as np
import torch


def tanimoto_one_to_many(query: torch.Tensor, fps: torch.Tensor) -> torch.Tensor:
    a = query.sum()
    b = fps.sum(dim=1)
    c = torch.matmul(fps, query)
    return c / (a + b - c + 1e-8)

def tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
    """
    Calculate Tanimoto similarity between fp1 and fp2.
    fp1: (N, D) float32, each row corresponds a fingerprint
    fp2: (N, D) float32

    return: (N,)
    """
    assert fp1.shape == fp2.shape, "fp1 and fp2 must have same shape"
    a = fp1.sum(axis=1)
    b = fp2.sum(axis=1)
    c = (fp1 * fp2).sum(axis=1)
    return c / (a + b - c + 1e-8)