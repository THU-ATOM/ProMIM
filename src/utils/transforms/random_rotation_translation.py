
from ._base import register_transform
import numpy as np
import torch

@register_transform('random_rotation_translation')
class RandomRotationTranslation(object):

    def __init__(self, shuffle=True, resolution='backbone+CB'):
        super().__init__()
        self.shuffle = shuffle
        assert resolution in ('full', 'backbone', 'backbone+CB')
        self.resolution = resolution

    def __call__(self, data):

        def random_rotation_matrix():
            a = torch.randn(3, 3)
            q, r = torch.linalg.qr(a)
            d = torch.diag(torch.sign(torch.diag(r)))
            q = torch.mm(q, d)
            if torch.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            return q
    
        if self.shuffle:
            R = random_rotation_matrix()
            t = torch.tensor(np.random.rand(3), dtype=torch.float)

            split_index = torch.where(data['chain_nb'] == 1)[0][0].item()
            chain0_pos_atoms = data['pos_atoms'][:split_index].clone()  # fix chain1
            t_chain0_pos_atoms = chain0_pos_atoms @ R + t
            pos_atoms = torch.cat([t_chain0_pos_atoms, data['pos_atoms'][split_index:]], dim=0)

            mask_atoms = data['mask_atoms'].unsqueeze(-1).repeat(1,1,3)
            mask_pos_atoms = torch.zeros_like(pos_atoms)
            data['transformed_pos_atoms'] = torch.where(mask_atoms==True, pos_atoms, mask_pos_atoms)

        else:
            if self.resolution == 'full':
                data['transformed_pos_atoms'] = data['transformed_pos_heavyatom'][:, :]
            elif self.resolution == 'backbone':
                data['transformed_pos_atoms'] = data['transformed_pos_heavyatom'][:, :4]
            elif self.resolution == 'backbone+CB':
                data['transformed_pos_atoms'] = data['transformed_pos_heavyatom'][:, :5]
        
        return data


@register_transform('random_rotation_translation_skempi')
class RandomRotationTranslationSkempi(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):

        def random_rotation_matrix():
            a = torch.randn(3, 3)
            q, r = torch.linalg.qr(a)
            d = torch.diag(torch.sign(torch.diag(r)))
            q = torch.mm(q, d)
            if torch.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            return q
    

        R = random_rotation_matrix()
        t = torch.tensor(np.random.rand(3), dtype=torch.float)

        pos_atoms = data['pos_atoms'].clone()
        for i, group_id in enumerate(data['group_id']):
            if group_id == 1:
                pos_atoms[i] = pos_atoms[i] @ R + t
            else: continue

        mask_atoms = data['mask_atoms'].unsqueeze(-1).repeat(1,1,3)
        mask_pos_atoms = torch.zeros_like(pos_atoms)
        data['transformed_pos_atoms'] = torch.where(mask_atoms==True, pos_atoms, mask_pos_atoms)
        
        return data