# CoBiSyn

## 📖 Introduction

This repository contains the official code and data for the paper **"CoBiSyn: A Bidirectional Search Framework for Chemical Synthesis Planning"**. We propose an effective search framework for chemical synthesis planning, which alternates between “backward decomposition” and “forward construction”,
while coordinating these two directions through shared frontier information.

![CoBiSyn](figs/framework.png)

## 📊 Usage

### Data Preparation

Download the relvent data and pre-trained models from [here](https://huggingface.co/CoBiSyn/CoBiSyn/tree/main) and put them in current directory. 

The files are organized as follows:
```
├── checkpoints
│   ├── dist.ckpt   # SynDistModel
│   ├── fwd.ckpt    # ForwardModel
│   ├── pretrain.ckpt   # Base RetroModel (without fine-tuned by triplets)
│   └── retro.ckpt  # RetroModel (fine-tuned by triplets)
└── dataset
    ├── benchmarks  # test cases
    │   ├── pistachio_hard_targets.txt  # Pistachio Reachable
    │   ├── pistachio_reachable_targets.txt # Pistachio Hard
    │   └── uspto190.pkl    # USPTO-190
    ├── raw_data/   # reactions and pathways
    ├── bbs_emb.index   # faiss index of pre-computed query embeddings (SynDistModel)
    ├── bbs_fps.h5  # pre-computed Morgan fingerprints of building blocks
    ├── bbs_fps.index   # faiss index of fingerprints (ForwardModel)
    ├── building-blocks.pkl     # building blocks
    └── index2template.json     # templates

```

### Environment Setup

```
conda env create -f environment.yml -n cobisyn
conda activate cobisyn
```

### Model Evaluation

Taking USPTO-190 as examples:
```
python test.py  \
    --test dataset/benchmarks/uspto190.pkl \
    --dump results/results_uspto190.pkl \
    --device cuda:0
```
The results will be saved in `dump`. Other parameters can be found in `CoBiSyn/args.py`.

### Find Synthesis Pathway

To `run` interface of `CoBiSyn` class is used to find synthesis pathway for molecule specified in SMILES format.

``` python
from CoBiSyn.cobisyn import CoBiSyn

target = 'SMILES of a molecule'
solver = CoBiSyn()  # init relevant parameters

route = solver.run(target)

print(route.finish())   # whether success
print(route.success_route())    # identified route
```

If you want to perform on your own building blocks, you first need to create the corresponding FAISS index using `scripts/build_fp_index.py` and `scripts/build_emb_index.py`, then repeat the above process.

### Single-Step Retrosynthesis

As describled in our paper, the retrosynthesis model used in CoBiSyn is first pre-trained on a large amount of USPTO reaction data (`dataset/raw_data/reactions-train.json.gz`), and then fine-tuned on the extracted triplets with conditional signals (`dataset/raw_data/retro-train-dataset.json`). If you would like to use our pre-trained model as a single-step retrosynthesis model, please proceed as follows: 
```python
from CoBiSyn.model.retro import RetroModel
from CoBiSyn.chem.mol import Molecule

model = RetroModel.load_from_checkpoint('checkpoints/pretrain.ckpt', mode='pretrain', map_loaction='cuda:0', strict=False)

mol = Molecule('SMILES of molecule', device=model.device)
model.predict_no_cond(mol)  # return top_k_scores and top_k_indices
```