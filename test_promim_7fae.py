import os
import copy
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm

from src.utils.misc import load_config, seed_all
from src.utils.data import PaddingCollate
from src.utils.train import *
from src.utils.transforms import Compose, SelectAtom, SelectedRegionFixedSizePatch
from src.utils.protein.parsers import parse_biopython_structure
from src.models.promim_ddg import DDG_Network
from pathlib import Path

class PMDataset(Dataset):

    def __init__(self, pdb_path, mutations):
        super().__init__()
        self.pdb_path = pdb_path

        self.data = None
        self.seq_map = None
        self._load_structure()

        self.mutations = self._parse_mutations(mutations)
        self.transform = Compose([
            SelectAtom('backbone+CB'),
            SelectedRegionFixedSizePatch('mut_flag', 128)
        ])

    
    def clone_data(self):
        return copy.deepcopy(self.data)

    def _load_structure(self):
        if self.pdb_path.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        elif self.pdb_path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError('Unknown file type.')

        structure = parser.get_structure(None, self.pdb_path)
        data, seq_map = parse_biopython_structure(structure[0])
        self.data = data
        self.seq_map = seq_map

    def _parse_mutations(self, mutations):
        parsed = []
        for m in mutations:
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])
            pos = (ch, seq, ' ')
            if pos not in self.seq_map: continue

            if mt == '*':
                for mt_idx in range(20):
                    mt = index_to_one(mt_idx)
                    if mt == wt: continue
                    parsed.append({
                        'position': pos,
                        'wt': wt,
                        'mt': mt,
                    })
            else:
                parsed.append({
                    'position': pos,
                    'wt': wt,
                    'mt': mt,
                })
        return parsed

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, index):
        data = self.clone_data()
        mut = self.mutations[index]
        mut_pos_idx = self.seq_map[mut['position']]

        data['mut_flag'] = torch.zeros(size=data['aa'].shape, dtype=torch.bool)
        data['mut_flag'][mut_pos_idx] = True
        data['aa_mut'] = data['aa'].clone()
        data['aa_mut'][mut_pos_idx] = one_to_index(mut['mt'])
        data = self.transform(data)
        data['ddG'] = 0
        data['mutstr'] = '{}{}{}{}'.format(
            mut['wt'],
            mut['position'][0],
            mut['position'][1],
            mut['mt']
        )

        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--outdir', type=str, default='./result_test_promim_7fae')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--idx_cvfolds', type=int, required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    seed_all(args.seed)

    # Data
    dataset = PMDataset(
        pdb_path = config.pdb,
        mutations = config.mutations,
    )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=PaddingCollate(), 
    )

    # Model
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    cv_mgr = CrossValidation(model_factory=DDG_Network, config=ckpt['config'], num_cvfolds=3)
    cv_mgr.load_state_dict(ckpt['model'])
    cv_mgr.to(args.device)

    result = []
    for batch in tqdm(loader):
        batch = recursive_to(batch, args.device)
        fold = args.idx_cvfolds
        model, _, _ = cv_mgr.get(fold)
        model.eval()  
        with torch.no_grad():
            _, out_dict = model(batch)
        for mutstr, ddG_pred in zip(batch['mutstr'], out_dict['ddG_pred'].cpu().tolist()):
            result.append({
                'mutstr': mutstr,
                'ddG_pred': ddG_pred,
            })
    result = pd.DataFrame(result)
    result = result.groupby('mutstr').mean().reset_index()
    result['rank'] = result['ddG_pred'].rank() / len(result)

    if not args.tag == '':
        all_results_name = 'all_' + args.tag + '.csv'
        interest_results_name = 'interest_' + args.tag + '.csv'
        interest_ranks_name = 'interest_ranks_' + args.tag + '.csv'
    else:
        all_results_name = 'all.csv'
        interest_results_name = 'interest.csv'
        interest_ranks_name = 'interest_ranks.csv'

    save_path_all = os.path.join(args.outdir, str(args.idx_cvfolds), all_results_name)
    Path(save_path_all).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(save_path_all)

    if 'interest' in config and config.interest:
        save_path_interest = os.path.join(args.outdir, str(args.idx_cvfolds), interest_results_name)
        Path(save_path_interest).parent.mkdir(parents=True, exist_ok=True)
        print(result[result['mutstr'].isin(config.interest)])
        df_interest = result[result['mutstr'].isin(config.interest)]
        df_interest.to_csv(save_path_interest, index=False)

        interest_rank_dict = {}
        for i in range(len(df_interest)):
            entry = df_interest.iloc[i]
            interest_rank_dict[entry['mutstr']] = entry['rank']
        df_interest_rank = pd.DataFrame([interest_rank_dict])
        save_path_interest_rank = os.path.join(args.outdir, str(args.idx_cvfolds), interest_ranks_name)
        Path(save_path_interest_rank).parent.mkdir(parents=True, exist_ok=True)
        df_interest_rank.to_csv(save_path_interest_rank, index=False)  

