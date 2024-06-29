import math
import torch
from torch.utils.data._utils.collate import default_collate

DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'aa_masked': 21,
    'aa_true': 21,
    'chain_nb': -1, 
    'pos14': 0.0,
    'chain_id': ' ', 
    'icode': ' ',
}


class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        return batch

class PPIRefPaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, max_length=64):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.max_length = max_length

    @staticmethod
    def _pad_last(x, n, split_idx, value=0):
        if isinstance(x, torch.Tensor):
            assert split_idx <= n and x.size(0)-split_idx <= n
            if split_idx == n and x.size(0)-split_idx == n:
                return x
            pad_size_1 = [n - split_idx] + list(x.shape[1:])
            pad_size_2 = [n - (x.size(0)-split_idx)] + list(x.shape[1:])
            pad_1 = torch.full(pad_size_1, fill_value=value).to(x)
            pad_2 = torch.full(pad_size_2, fill_value=value).to(x)
            return torch.cat([x[:split_idx], pad_1, x[split_idx:], pad_2], dim=0)
        elif isinstance(x, list):
            pad_1 = [value] * (n - split_idx)
            pad_2 = [value] * (n - (len(x)-split_idx))
            return x[:split_idx] + pad_1 + x[split_idx:] + pad_2
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n, split_idx):
        return torch.cat([
            torch.zeros([split_idx], dtype=torch.bool),
            torch.ones([n-split_idx], dtype=torch.bool),
            torch.zeros([l-split_idx], dtype=torch.bool),
            torch.ones([n-(l-split_idx)], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):

        split_idxes = [torch.where(data['chain_nb'] == 1)[0][0].item() for data in data_list]
        keys = self._get_common_keys(data_list)
        data_list_padded = []
        for i, data in enumerate(data_list):
            data_padded = {
                k: self._pad_last(v, self.max_length, split_idx=split_idxes[i], value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), self.max_length, split_idxes[i])
            data_list_padded.append(data_padded)

        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        return batch
