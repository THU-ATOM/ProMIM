import sys
sys.path.append('./PPIRef')
from ppiref.utils.misc import download_from_zenodo
from ppiref.split import read_fold
import os
from tqdm import tqdm
import random
import pickle


def check_path_exist(path):
    full_id = path.split('/')[-1][:-4]
    partner_1 = full_id.split('_')[1]
    partner_2 = full_id.split('_')[2]
    case_1 = path[:-8] + f'_{partner_1.upper()}_{partner_2.upper()}.pdb'
    case_2 = path[:-8] + f'_{partner_1.lower()}_{partner_2.lower()}.pdb'
    case_3 = path[:-8] + f'_{partner_1.upper()}_{partner_2.lower()}.pdb'
    case_4 = path[:-8] + f'_{partner_1.lower()}_{partner_2.upper()}.pdb'
    if os.path.exists(case_1):
        return case_1
    elif os.path.exists(case_2):
        return case_2
    elif os.path.exists(case_3):
        return case_3
    elif os.path.exists(case_4):
        return case_4
    else:
        return False
    

if __name__ == '__main__':
    # Download PPIRef
    download_from_zenodo('ppi_6A.zip')

    # Read Paths of PPIRef50K
    fold = read_fold('ppiref_6A_filtered_clustered_04', 'whole', full_paths=True)
    failed_path = []
    ppiref50k_path = []
    for path in tqdm(fold, desc='Checking Path Existence'):
        fixed_path = check_path_exist(str(path))
        if not fixed_path:
            failed_path.append(path)
        else:
            ppiref50k_path.append(str(fixed_path))
    if not len(failed_path) == 0:
        print('Failed Paths:', len(failed_path))
    else:
        print('All Paths Exist')

    # Split PPIRef50K
    seed = 42
    train_ratio = 0.995
    random.seed(seed)
    random.shuffle(ppiref50k_path)
    ppiref50k_path_train = ppiref50k_path[:int(train_ratio * len(ppiref50k_path))]
    ppiref50k_path_val = ppiref50k_path[int(train_ratio * len(ppiref50k_path)):]

    print('Training Size:', len(ppiref50k_path_train))
    print('Validation Size:', len(ppiref50k_path_val))

    with open('./PPIRef/ppiref/data/ppiref/ppiref50k_path_train', 'wb') as f:
        pickle.dump(ppiref50k_path_train, f)
    with open('./PPIRef/ppiref/data/ppiref/ppiref50k_path_val', 'wb') as f:
        pickle.dump(ppiref50k_path_val, f)