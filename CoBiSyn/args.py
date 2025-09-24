import argparse

parser = argparse.ArgumentParser()

# Models
parser.add_argument("--retro_model", type=str, default='checkpoints/retro.ckpt')
parser.add_argument("--fwd_model", type=str, default='checkpoints/fwd.ckpt')
parser.add_argument("--dist_model", type=str, default='checkpoints/dist.ckpt')

# Dataset
parser.add_argument("--building_blocks", type=str, default='dataset/building-blocks.pkl')
parser.add_argument("--idx2temp", type=str, default='dataset/index2template.json')
parser.add_argument("--test", type=str)


parser.add_argument("--dist_index", type=str, default='dataset/bbs_emb.index')
parser.add_argument("--bbs_fps", type=str, default='dataset/bbs_fps.h5')
parser.add_argument("--bbs_fps_index", type=str, default='dataset/bbs_fps.index')

parser.add_argument("--dump", type=str)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--iterations", type=int, default=500)


args = parser.parse_args()