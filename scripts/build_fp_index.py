import h5py, os
import pickle
import argparse
import faiss
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def compute_fp_for_smiles_list(smiles_list):
    out = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    valid_index = []
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    for i, s in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            bv = generator.GetFingerprint(mol)
            onbits = list(bv.GetOnBits())
            for b in onbits:
                out[i, b] = 1
            valid_index.append(i)
        except Exception:
            continue
    return out, valid_index

def packbits_rows(bitmatrix):
    # bitmatrix: shape (n, NBITS), dtype=uint8 (0/1)
    packed = np.packbits(bitmatrix, axis=1)
    return packed  # shape (n, NBITS//8) dtype=uint8

def worker_compute(smiles_chunk):
    # receives list of SMILES strings, returns packed np.uint8 array
    bitmat, _ = compute_fp_for_smiles_list(smiles_chunk)
    packed = packbits_rows(bitmat)
    return packed

def chunked_iterable(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

def iter_h5_in_batches(h5_path: str, dataset: str, batch_size: int) -> Iterator[np.ndarray]:
    """Yield packed uint8 arrays from HDF5 in batches.
    Each yielded array shape: (b, BYTES_PER_VEC), dtype=uint8
    """
    with h5py.File(h5_path, "r") as f:
        dset = f[dataset]
        N = dset.shape[0]
        for i in range(0, N, batch_size):
            chunk = dset[i: i + batch_size]
            yield chunk

def unpack_uint8_to_float32(packed: np.ndarray) -> np.ndarray:
    """Convert packed uint8 bit-arrays (shape (n, BYTES_PER_VEC)) to
    float32 0/1 arrays (shape (n, D)). Uses np.unpackbits which is fast.
    """
    # np.unpackbits returns dtype=uint8; convert to float32
    bits = np.unpackbits(packed, axis=1)
    return bits.astype("float32")

parser = argparse.ArgumentParser()

parser.add_argument("--building_blocks", type=str, default='dataset/building-blocks.pkl')
parser.add_argument("--fps_dump", type=str, default='dataset/bbs_fps.h5')
parser.add_argument("--index_dump", type=str)

parser.add_argument("--nlist", type=int, default=4096)
parser.add_argument("--m", type=int, default=32)
parser.add_argument("--nbits", type=int, default=8)
parser.add_argument("--train_size", type=int, default=1_000_000)
parser.add_argument("--batch_size", type=int, default=100_000)
parser.add_argument("--use_float16", type=bool, default=True)




if __name__ == '__main__':
    args = parser.parse_args()

    OUTPUT_H5 = args.fps_dump
    NBITS = 2048
    CHUNK_SIZE = 100000
    DATASET_NAME = "bbs_fps"    # dataset inside the h5 file; shape (N, 256), dtype=uint8
    BYTES_PER_VEC = NBITS // 8  # 256 for 2048-bit
    BATCH_SIZE = args.batch_size
    TRAIN_SAMPLES = args.train_size 
    NLIST = args.nlist      # number of coarse clusters for IVF
    M_PQ = args.m           # number of subquantizers in PQ
    BITS_PER_CODE = 8   
    GPU_ID = 0

    USE_FLOAT16_GPU = args.use_float16  # store GPU index in float16 to save memory
    INDEX_OUTPUT = args.index_dump

    D = NBITS # dimensionality after unpackbits
    assert D % M_PQ == 0, "D must be divisible by M_PQ"

    # calculate fingerprints
    print(f'Loading building blocks from {args.building_blocks}')
    with open(args.building_blocks, 'rb') as f:
        bbs = pickle.load(f)

    if os.path.exists(OUTPUT_H5):
        print("Opening existing HDF5 (append mode)")
        h5f = h5py.File(OUTPUT_H5, 'a')
        if DATASET_NAME in h5f:
            dset = h5f[DATASET_NAME]
        else:
            dset = h5f.create_dataset(DATASET_NAME, shape=(0, NBITS//8),
                                    maxshape=(None, NBITS//8),
                                    dtype='uint8', chunks=(min(CHUNK_SIZE, 10000), NBITS//8),
                                    compression="lzf")
    else:
        h5f = h5py.File(OUTPUT_H5, 'w')
        dset = h5f.create_dataset(DATASET_NAME, shape=(0, NBITS//8),
                                maxshape=(None, NBITS//8),
                                dtype='uint8', chunks=(min(CHUNK_SIZE, 10000), NBITS//8),
                                compression="lzf")
        
    start_index = dset.shape[0]
    bbs_remaining = bbs[start_index:] 

    with Pool(processes=cpu_count() // 4) as pool:
        for packed in tqdm(
            pool.imap(worker_compute, chunked_iterable(bbs_remaining, CHUNK_SIZE)),
            total=(len(bbs_remaining) + CHUNK_SIZE - 1) // CHUNK_SIZE,
            desc="processing/writing"
        ):
            n = packed.shape[0]
            if n == 0:
                continue
            old = dset.shape[0]
            dset.resize(old + n, axis=0)
            dset[old:old+n, :] = packed

    total_written = dset.shape[0]
    h5f.close()
    print("Calculating fingerprints done! Total written:", total_written)


    # build index
    with h5py.File(OUTPUT_H5, "r") as f:
        all_packed = f[DATASET_NAME][:]
        N = len(all_packed)
    rng = np.random.default_rng(42)
    train_idx = rng.choice(all_packed.shape[0], TRAIN_SAMPLES, replace=False)
    packed_train = all_packed[train_idx]
    train_vectors = unpack_uint8_to_float32(packed_train)
    print(f"Training vectors ready: {train_vectors.shape}, dtype={train_vectors.dtype}")

    print("Building IVFPQ index...")
    quantizer = faiss.IndexFlatIP(D)
    index_cpu = faiss.IndexIVFPQ(quantizer, D, NLIST, M_PQ, BITS_PER_CODE, faiss.METRIC_INNER_PRODUCT)
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    print("Training index on gpu...")
    index_gpu.train(train_vectors)
    print("Training finished.")

    with h5py.File(OUTPUT_H5, "r") as f:
        dset = f[DATASET_NAME]
        for i in range(0, N, BATCH_SIZE):
            packed = dset[i: i + BATCH_SIZE]
            xb = unpack_uint8_to_float32(packed)
            index_gpu.add(xb)
            print(f"Added vectors {i} -- {i + xb.shape[0]} to GPU index. current ntotal={index_gpu.ntotal}")

    index_cpu = faiss.index_gpu_to_cpu(index_gpu)
    faiss.write_index(index_cpu, INDEX_OUTPUT)
    print(f"Index is saved in {INDEX_OUTPUT}")