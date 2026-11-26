import copy
import logging

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .tools import apply_selection, build_new_variables, extract_fields_from_expr
from .fileio import read_files
from .config import DataConfig

_logger = logging.getLogger("SynapseLogger")

def prepare_data(data: ak.Array, data_cfg: DataConfig, dataset_type: str) -> dict:
    """
    Prepare the data according to the configuration.
    """
    # Apply selection
    data = apply_selection(data, data_cfg.selection, data_cfg.new_variables)
    # Build new variables
    data = build_new_variables(data, data_cfg.new_variables, supress_warning=True)
    # TODO: implement padding for variable length arrays
    # TODO: check if the data is empty after selection
    # TODO: check if the data only have one class in classification tasks
    # TODO: check if the data have entries with more/less than one active label in classification tasks
    # Check if all active keys are valid
    # active_keys: inputs, label_keys, weights
    # spectators don't need to be checked, since they will not be fed to the model
    for var in data_cfg.active_keys:
        if np.any(np.isnan(data[var])):
            data[var] = np.nan_to_num(data[var], posinf=np.inf, neginf=-np.inf)
            _logger.warning(f"NaN values found in variable '{var}', replaced with zeros.")
        if np.any(np.isinf(data[var])):
            raise RuntimeError(f"Infinite values found in variable '{var}'.") # FIXME: should be handled more gracefully?
    # Convert to numpy arrays, otherwise we need to define a complex collate function
    prepared_data = {}
    for k, vs in data_cfg.inputs.items():
        dtype = np.int8 if k == "relation_info" else np.float32
        prepared_data[k] = ak.to_numpy(ak.concatenate([data[v][:, np.newaxis] for v in vs], axis=1)).astype(dtype)
    for k in data_cfg.final_label_keys:
        if k == '_class_label':
            prepared_data[k] = ak.to_numpy(data[k]).astype(np.int64)
        else:
            prepared_data[k] = ak.to_numpy(data[k])
    if data_cfg.weight_key in data.fields:
        prepared_data[data_cfg.weight_key] = ak.to_numpy(data[data_cfg.weight_key]).astype(np.float32)
    if dataset_type == 'test':
        for k in data_cfg.spectators:
            prepared_data[k] = ak.to_numpy(data[k])

    return prepared_data

# TODO: implement a hybrid dataset that combines map-style and iterable-style datasets or simply an iterable dataset
# class HybridDataset(IterableDataset):
#     def __init__(self, file_list):
#         self.file_list = file_list
#
#     def __iter__(self):
#         for file in self.file_list:
#             chunk_data = load_chunk(file)
#
#             map_dataset = MapStyleDataset(chunk_data)
#             yield from map_dataset

# TODO: decide whether to support lazy loading or not
class MapStyleDataset(Dataset):
    def __init__(self, file_list: list[str], dataset_type: str, data_cfg: DataConfig):
        self.dataset_type = dataset_type
        self.data_cfg = data_cfg
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid dataset type: {dataset_type}. Must be one of ['train', 'val', 'test'].")
        keys = data_cfg.active_keys
        if dataset_type == 'test':
            keys.extend(data_cfg.spectators)
        if data_cfg.new_variables:
            for expr in data_cfg.new_variables.values():
                keys.extend(extract_fields_from_expr(expr))
        load_range = tuple(data_cfg.get(f'{dataset_type}_load_range', (0.0,1.0)))
        normalized_range = data_cfg.get(f'{dataset_type}_normalized_range', True)
        raw_data, _ = read_files(file_list, list(set(keys)), load_range, normalized_range, merge=True, tree_name=data_cfg.get('tree_name'))
        self.data = prepare_data(raw_data, data_cfg, dataset_type)

        lengths = [len(arr) for arr in self.data.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All feature arrays must have the same length")
        self.num_samples = lengths[0]

    @property
    def mean_weight(self) -> float:
        weights = self.data[self.data_cfg.weight_key]
        if weights.size == 0:
            raise ValueError("Cannot compute the mean training weight for an empty fold.")
        mean = float(weights.mean(dtype=np.float64))
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError(f"Mean training weight must be finite and positive, got {mean}.")
        return mean

    def get_input_norm_dict(self):
        chunk_size = self.data_cfg.get("normalization_chunk_size")
        if chunk_size:
            return self._chunked_input_norm(int(chunk_size))
        # (num_samples, num_evt_feats)
        ary_evt_feats = torch.tensor(self.data.get('evt_feats'))
        # (num_samples,)
        weight_ary = torch.tensor(self.data.get(self.data_cfg.weight_key, np.ones(self.num_samples)))
        # (num_samples, num_obj_feats, num_obj)
        ary_obj_feats = torch.tensor(self.data.get('obj_feats'))
        # (num_samples, 1, num_obj)
        ary_mask = torch.tensor(self.data.get('mask'))

        norm_dict = {}
        # norm for evt_feats. Shape: (num_samples, num_evt_feats) -> (num_evt_feats,)
        ary_evt_weights = weight_ary.unsqueeze(-1) # (num_samples, 1)
        weighted_mean = torch.sum(ary_evt_feats * ary_evt_weights, dim=0) / torch.sum(ary_evt_weights)
        weighted_std = torch.sqrt(torch.sum(ary_evt_weights * (ary_evt_feats - weighted_mean)**2, dim=0) / torch.sum(ary_evt_weights))
        norm_dict['evt_feats'] = {'mean': weighted_mean, 'std': weighted_std.clip(min=1e-6)}


        # norm for obj_feats. Shape: (num_samples, num_obj_feats, num_obj) -> (num_obj_feats,)
        ary_obj_feats = ary_obj_feats.permute(0, 2, 1) # (num_samples, num_obj, num_obj_feats)
        ary_obj_weights = ary_mask.view(ary_obj_feats.shape[0], -1) * weight_ary.unsqueeze(-1)  # (num_samples, num_obj)
        ary_obj_weights = ary_obj_weights.unsqueeze(-1)
        sum_weights = torch.sum(ary_obj_weights)
        weighted_mean = torch.sum(ary_obj_feats * ary_obj_weights, dim=(0,1)) / sum_weights
        weighted_std = torch.sqrt(torch.sum(ary_obj_weights * (ary_obj_feats - weighted_mean)**2, dim=(0,1)) / sum_weights)
        norm_dict['obj_feats'] = {'mean': weighted_mean, 'std': weighted_std.clip(min=1e-6)}

        return norm_dict

    def _chunked_input_norm(self, chunk_size):
        if chunk_size < 1:
            raise ValueError("Normalization chunks must be positive")
        globals_ = self.data["evt_feats"]
        objects = self.data["obj_feats"]
        valid = self.data["mask"][:, 0, :]
        weights = self.data[self.data_cfg.weight_key]
        sums = [np.zeros(globals_.shape[1]), np.zeros(objects.shape[1])]
        totals = np.zeros(2)
        for start in range(0, len(weights), chunk_size):
            chunk = slice(start, start + chunk_size)
            w = weights[chunk].astype(np.float64)
            ow = w[:, None] * valid[chunk]
            totals += (w.sum(), ow.sum())
            sums[0] += np.einsum("n,nf->f", w, globals_[chunk])
            sums[1] += np.einsum("np,nfp->f", ow, objects[chunk])
        if not np.all(np.isfinite(totals) & (totals > 0)):
            raise ValueError("Normalization requires positive finite event and object weight")
        means = [total / weight for total, weight in zip(sums, totals)]
        variances = [np.zeros_like(mean) for mean in means]
        for start in range(0, len(weights), chunk_size):
            chunk = slice(start, start + chunk_size)
            w = weights[chunk].astype(np.float64)
            ow = w[:, None] * valid[chunk]
            variances[0] += np.einsum("n,nf->f", w, np.square(globals_[chunk] - means[0]))
            variances[1] += np.einsum("np,nfp->f", ow, np.square(objects[chunk] - means[1][:, None]))
        return {key: {"mean": torch.tensor(mean, dtype=torch.float32),
                      "std": torch.tensor(np.maximum(np.sqrt(v / total), 1.e-6), dtype=torch.float32)}
                for key, mean, v, total in zip(("evt_feats", "obj_feats"), means, variances, totals)}



    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Don't use np.ndarray.copy() here, as it will return a view of the original array in some cases
        # TODO: copy.deepcopy() is not efficient, consider using a more efficient method if needed
        x = {k: copy.deepcopy(self.data[k][idx]) for k in self.data_cfg.inputs.keys()}
        y = {k: copy.deepcopy(self.data[k][idx]) for k in self.data_cfg.final_label_keys}
        w = {self.data_cfg.weight_key:  copy.deepcopy(self.data[self.data_cfg.weight_key][idx])}
        s = {}
        if self.dataset_type == 'test':
            # The default collator preserves uint64 event IDs when given zero-dimensional arrays.
            s = {k: np.asarray(self.data[k][idx]) for k in self.data_cfg.spectators}
        return x, y, w, s
