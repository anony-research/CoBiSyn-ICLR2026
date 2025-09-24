import gzip, json, pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
import torch
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("train")


if __name__ == '__main__':
    file = 'validate'
    mode = 'finetune'

    if mode == 'finetune':
        with open(f'dataset/rdchiral/retro-{file}-dataset.json', 'r') as f:
            rxns = json.load(f)
    else:
        with gzip.open(f'dataset/rdchiral/{file}.json.gz') as f:
            rxns = json.load(f)

    fps = []
    labels = []
    generator = GetMorganGenerator(radius=2, fpSize=2048)

    for item in tqdm(rxns):
        if mode == 'finetune':
            mol_fp = torch.tensor(
                list(generator.GetFingerprint(Chem.MolFromSmiles(item['molecule']))), 
                dtype=torch.float32
            )
            cond_fp = torch.tensor(
                list(generator.GetFingerprint(Chem.MolFromSmiles(item['condition']))),
                dtype=torch.float32
            )
            fp = torch.cat([mol_fp, cond_fp], dim=0)
            fps.append(fp)
            labels.append(item['template'])
        
        else:
            fp = generator.GetFingerprint(Chem.MolFromSmiles(item['products']))
            arr = np.array(fp) 
            fps.append(arr)
            labels.append(item["template_id"])

    fps = np.array(fps, dtype=np.uint8)         # (N, 2048)
    labels = np.array(labels, dtype=np.int32)   # (N,)

    if mode == 'finetune':
        with open(f'dataset/rdchiral/retro-{file}-dataset.pkl', 'wb') as f:
            pickle.dump((fps, labels), f)
    else:
        np.save(f"dataset/rdchiral/{file}-fingerprints.npy", fps)
        np.save(f"dataset/rdchiral/{file}-labels.npy", labels)
