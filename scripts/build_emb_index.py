import h5py
import numpy as np
import torch
import argparse
import faiss
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from CoBiSyn.model.syndist import SynDistModel

class MorganDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5file = h5py.File(h5_path, 'r')
        self.fp_data = self.h5file['bbs_fps']  # shape=(23_000_000, 256), dtype=uint8

    def __len__(self):
        return self.fp_data.shape[0]

    def __getitem__(self, idx):
        packed = self.fp_data[idx]  # 256 bytes
        bits = np.unpackbits(packed)  # shape=(2048,)
        return torch.from_numpy(bits).float()
    

def calculate_embeddings(args):
    dataset = MorganDataset(args.bbs_fps)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=32)

    model = SynDistModel.load_from_checkpoint(args.dist_model, map_location=args.device)
    emb_dim = model.hparams['dim_emb']  

    with h5py.File(args.dump, "a") as f:
        if "embeddings" not in f:
            emb_ds = f.create_dataset(
                "embeddings", 
                shape=(0, emb_dim),
                maxshape=(None, emb_dim), 
                dtype="float32",            
                chunks=(1024, emb_dim),     
                compression="gzip",         
                compression_opts=4 
            )
            idx = 0
        else:
            emb_ds = f["embeddings"]
            idx = emb_ds.shape[0]

        print(f"Starting from index {idx}/{len(dataset)}")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, initial=idx // dataloader.batch_size)):
                if idx >= (i+1) * dataloader.batch_size:
                    continue

                batch = batch.to(model.device)
                emb = model.cal_candidate_emb(batch).cpu().numpy()  # shape=(batch_size, emb_dim)
                
                emb_ds.resize((idx + len(emb), emb_dim))
                emb_ds[idx:idx + len(emb)] = emb
                idx += len(emb)
                
                if i % 100 == 0:
                    f.flush()
                    print(f"[Flush] {idx} data points written.")
    print("Embedding calculation done!")


parser = argparse.ArgumentParser()

parser.add_argument("--dist_model", type=str, default='checkpoints/dist.ckpt')
parser.add_argument("--bbs_fps", type=str)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--emb_dump", type=str, default='dataset/bbs_embeddings.h5')
parser.add_argument("--index_dump", type=str)

# index config
parser.add_argument("--nlist", type=int, default=4096)
parser.add_argument("--m", type=int, default=32)
parser.add_argument("--nbits", type=int, default=8)
parser.add_argument("--train_size", type=int, default=1_000_000)
parser.add_argument("--batch_size", type=int, default=100_000)



if __name__ == "__main__":
    args = parser.parse_args()

    calculate_embeddings(args)
    
    nlist = args.nlist
    m = args.m
    nbits = args.nbits
    train_size = args.train_size

    print("Loading embeddings for training...")
    with h5py.File(args.emb_dump) as h5file:
        embds = h5file['embeddings']
        N, d = embds.shape

        idx = np.random.choice(N, train_size, replace=False)
        idx.sort()
        xb_train = embds[:train_size].astype("float32")
    print("Embeddings loaded.")

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 1, index)

    print("Training the index...")
    index.train(xb_train)

    batch_size = args.batch_size
    with h5py.File(args.emb_dump, "r") as f:
        dataset = f["embeddings"]
        nb = dataset.shape[0]

        for i in range(0, nb, batch_size):
            xb = dataset[i:i+batch_size].astype('float32')
            index.add(xb)
            print(f"Added {i+len(xb)} / {nb} vectors")

    cpu_index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(cpu_index, args.index_dump)
    print(f'Index is saved in {args.index_dump}')