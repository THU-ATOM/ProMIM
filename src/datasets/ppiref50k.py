import sys
sys.path.append('.')
import os
import math
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
from src.utils.protein.parsers import parse_ppiref_structure
import Bio.PDB as PDB


def _process_interface(pdb_path, interface_id):
    parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    try:
        interface = parser.get_structure(interface_id, pdb_path)[0]
    except:
        print(f'[INFO] Failed to parse interface: {interface_id}')
        return None

    data, _ = parse_ppiref_structure(interface, missing_threshold=0.7)
    if data is None:
        print(f'[INFO] Failed to parse interface. Too few valid residues: {pdb_path}')
        return None
    data['id'] = interface_id
    return data

class PPIRef50KDataset(Dataset):

    MAP_SIZE = 384*(1024*1024*1024) # 384GB

    def __init__(
        self, 
        split,
        splits_path,
        processed_dir = './data/PPIRef50K_processed',
        num_preprocess_jobs = math.floor(cpu_count() * 0.8),
        transform = None,
        reset = False,
    ):
        super().__init__()

        self.split = split
        self.splits_path = splits_path
        self.processed_dir = os.path.join(processed_dir, split)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.num_preprocess_jobs = num_preprocess_jobs
        self.transform = transform

        self.db_conn = None
        self.db_keys = None
        self.load_interfaces(reset)

    @property
    def processed_lmdb_path(self):
        return os.path.join(self.processed_dir,'interfaces.lmdb')
    
    @property
    def keys_path(self):
        return os.path.join(self.processed_dir, 'keys.pkl')    


    def load_interfaces(self, reset=False):
        if os.path.exists(self.processed_lmdb_path) and not reset:
            return
        with open(self.splits_path, 'rb') as f:
            self.dataset = pickle.load(f)

        tasks = []
        for pdb_path in tqdm(self.dataset):
            interface_id = pdb_path.split('/')[-1][:-4]
            tasks.append(
                delayed(_process_interface)(pdb_path, interface_id)
            )

        # Split data into chunks
        chunk_size = 8192
        task_chunks = [
            tasks[i*chunk_size:(i+1)*chunk_size] 
            for i in range(math.ceil(len(tasks)/chunk_size))
        ]        

        # Establish database connection
        db_conn = lmdb.open(
            self.processed_lmdb_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )

        keys = []
        total_count = 0
        for i, task_chunk in enumerate(task_chunks):
            with db_conn.begin(write=True, buffers=True) as txn:
                processed = Parallel(n_jobs=self.num_preprocess_jobs)(
                    task
                    for task in tqdm(task_chunk, desc=f"Chunk {i+1}/{len(task_chunks)}")
                )
                stored = 0
                for data in processed:
                    if data is None:
                        continue
                    key = str(total_count).encode()
                    keys.append(key)
                    txn.put(key=key, value=pickle.dumps(data))
                    total_count += 1
                    stored += 1
                print(f"[INFO] {stored} processed for chunk#{i+1}")
        print(f"[INFO] {total_count} processed for all chunk")
        db_conn.close()    

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)            

    def _connect_db(self):
        assert self.db_conn is None
        self.db_conn = lmdb.open(
            self.processed_lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with open(self.keys_path, 'rb') as f:
            self.db_keys = pickle.load(f)

    def _close_db(self):
        self.db_conn.close()
        self.db_conn = None
        self.db_keys = None

    def _get_from_db(self, index):
        if self.db_conn is None:
            self._connect_db()
        data = pickle.loads(self.db_conn.begin().get(str(index).encode()))
        return data

    def __len__(self):
        with open(self.keys_path, 'rb') as f:
            keys = pickle.load(f)  
        return len(keys)
    
    def __getitem__(self, index):

        data = self._get_from_db(index)

        if self.transform is not None:
            data = self.transform(data)

        return data

def get_PPIRef50K_dataset(cfg):
    from src.utils.transforms import get_transform
    return PPIRef50KDataset(
        split=cfg.split,
        splits_path = cfg.splits_path,
        processed_dir = cfg.processed_dir,
        transform = get_transform(cfg.transform),
    )

if __name__ == '__main__':
    
    train_dataset = PPIRef50KDataset(
        split='train',
        splits_path = './PPIRef/ppiref/data/ppiref/ppiref50k_path_train',
        processed_dir = './data/PPIRef50K_processed',
    )

    val_dataset = PPIRef50KDataset(
        split='val',
        splits_path = './PPIRef/ppiref/data/ppiref/ppiref50k_path_val',
        processed_dir = './data/PPIRef50K_processed',
    )