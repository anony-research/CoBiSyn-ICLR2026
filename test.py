from __future__ import annotations
import os
import pickle, json, ast
from tqdm import tqdm


from CoBiSyn.args import args
from CoBiSyn.cobisyn import CoBiSyn
from CoBiSyn.chem.mol import Molecule
from CoBiSyn.model.retro import RetroModel
from CoBiSyn.model.forward import ForwardModel
from CoBiSyn.model.syndist import SynDistModel

def read_test_target(filename):
    ext =  os.path.splitext(filename)[1].lower()
    if ext == '.pkl':
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        for data in dataset:
            yield data[0].split('>')[0]
    elif ext == '.txt':
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    yield ast.literal_eval(line.strip())[0]
    else:
        raise NotImplementedError


def test():
    device = args.device

    print('Loading building blocks...')
    with open(args.building_blocks, 'rb') as f:
        bbs = pickle.load(f)
    bbs_mols = [Molecule(m, device=device) for m in bbs]
    bbs_set = set(bbs_mols)
    print(f"A total of {len(bbs)} building blocks.")

    print('Loading templates...')
    with open(args.idx2temp, 'r') as f:
        idx2temp = json.load(f)
        idx2temp = {int(i): temp for i, temp in idx2temp.items()}

    print('Loading test cases...')
    test_targets = read_test_target(args.test)


    print('Loading models...')
    retro: RetroModel = RetroModel.load_from_checkpoint(args.retro_model, mode='finetune', map_location=device, strict=False)
    fwd: ForwardModel = ForwardModel.load_from_checkpoint(args.fwd_model, map_location=device, strict=False)
    syndist: SynDistModel = SynDistModel.load_from_checkpoint(args.dist_model, map_location=device)
    fwd.load_ref_fps(fps_path=args.bbs_fps, faiss_path=args.bbs_fps_index)
    syndist.load_index(args.dist_index)
    retro.eval()
    fwd.eval()
    syndist.eval()


    solver = CoBiSyn(
        bbs_mols=bbs_mols,
        bbs_set=bbs_set,
        idx2temp=idx2temp,
        retro=retro,
        fwd=fwd,
        syndist=syndist
    )
    
    results = []
    pbar = tqdm(test_targets)
    for target in pbar:
        pbar.set_description(f"Finding routes for {target}")
        try:
            route = solver.run(target, args.iterations)
            results.append({
                'succ': route.finish(),
                'route': route.success_route() if route.finish() else None,
                'iterations': route.round,
                'time': route.time
            })
        except Exception as e:
            results.append({
                'succ': False,
                'route': None,
                'iterations': None,
                'time': None
            })
    
    with open(args.dump, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results are saved in {args.dump}')


if __name__ == '__main__':
    test()