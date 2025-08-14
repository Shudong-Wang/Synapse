# a script to convert the CAF flat ntuple to the format suitable for Synapse
import argparse
import glob
import math
import os
from pathlib import Path

import awkward as ak
import yaml
from tqdm import tqdm
import gc
import psutil


from synapse.core.fileio import read_files, write_file
from synapse.core.tools import build_new_variables

def valid_config(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path

def load_config(config_path) -> dict:
    """
    Load the configuration file and extract the branches to be used

    Args:
        config_path (str): path to the configuration file
    Returns:
        dict: configuration dictionary with branches
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    branches = set()
    for obj_name in cfg.get('object_names', []):
        for var in cfg.get('object_variables', []):
            branches.add(f'{obj_name}_{var}')

    for var in cfg.get('event_variables', []):
        branches.add(var)

    cfg['branches'] = list(branches)

    return cfg


def convert(input_data: ak.Array, cfg: dict):
    """
    Convert the input flat ntuple with memory limit and a single progress bar for overall progress.
    """
    # Memory limit: Use 50% of available RAM
    memory_limit = int(psutil.virtual_memory().available * 0.5)
    object_names = cfg.get('object_names', [])
    object_variables = cfg.get('object_variables', [])
    sample_size = min(1000, len(input_data))
    sample_data = input_data[:sample_size]
    total_size = 0

    # Estimate memory usage
    for feat in object_variables:
        arrays = [sample_data[f"{obj_name}_{feat}"] for obj_name in object_names]
        concatenated = ak.concatenate([arr[:, None] for arr in arrays], axis=1)
        total_size += concatenated.nbytes
    avg_row_size = total_size / sample_size
    chunk_size = max(1, int(memory_limit / avg_row_size))

    output_data = {}

    total_chunks = (len(input_data) + chunk_size - 1) // chunk_size

    # Total steps: number of features + number of event variables + 1 (for zipping and new variables)
    total_steps = (
        len(object_variables) * total_chunks +
        len(cfg.get('event_variables', [])) +
        (1 if 'new_variables' in cfg else 0)
    )

    with tqdm(total=total_steps, desc="Converting data", unit="step") as pbar:

        for feat in object_variables:
            feat_data = []
            for i in range(0, len(input_data), chunk_size):
                chunk_data = input_data[i:i + chunk_size]
                arrays = [chunk_data[f"{obj_name}_{feat}"] for obj_name in object_names]
                concatenated = ak.concatenate([arr[:, None] for arr in arrays], axis=1)
                feat_data.append(concatenated)
                del chunk_data
                gc.collect()
                pbar.update(1)  # Update progress bar for each chunk
            output_data[feat] = ak.concatenate(feat_data, axis=0)

        for feat in cfg.get('event_variables', []):
            output_data[feat] = input_data[feat]
            pbar.update(1)  # Update progress bar for each event variable

        output_data = ak.zip(output_data, depth_limit=1)

        if 'new_variables' in cfg:
            output_data = build_new_variables(output_data, cfg.get('new_variables'))
            pbar.update(1)  # Update progress bar for new variables


    return output_data





def split_folds(input_data: ak.Array, cfg: dict, fold_splitting_var: str) -> list[ak.Array]:
    """
    Split the data into folds
    """
    remainders = input_data[fold_splitting_var] % cfg.get('k_folds', 1)
    folds = [input_data[remainders == r] for r in range(cfg.get('k_folds', 1))]

    return folds

def main():
    parser = argparse.ArgumentParser(description="Convert ROOT file")
    parser.add_argument('-c','--config', type=str, required=True, help='Configuration file path')

    args = parser.parse_args()

    print("Starting hhml CAF ntuple conversion process...")

    config = load_config(valid_config(args.config))

    in_file_paths = []
    for file_path in config.get('in_file_paths', []):
        in_file_paths.extend(glob.glob(file_path))

    print("Converting...")

    data_in, file_names_in = read_files(file_paths=in_file_paths,
                                        keys=config['branches'],
                                        merge=config['merge_input'],
                                        tree_name=config.get('in_tree_name'))

    if config.get('merge_input'):
        data_out = convert(data_in, config)
        if config.get('k_folds', 1) > 1:
            data_out_folds = split_folds(data_out, config, config.get('fold_splitting_var', 'eventNumber'))
            for i, fold in enumerate(data_out_folds):
                file_path_out = os.path.join(config['output_dir'], "merged" ,f"fold_{i}.root")
                write_file(file_path_out, fold, tree_name=config.get('out_tree_name', 'tree'))
        else:
            file_path_out = os.path.join(config['output_dir'], "merged" ,f"merged_total.root")
            write_file(file_path_out, data_out, tree_name=config.get('out_tree_name', 'tree'))
    else:
        data_out = []
        for data in data_in:
            data_out.append(convert(data, config))
        if config.get('k_folds', 1) > 1:
            for i, data in enumerate(data_out):
                data_folds = split_folds(data, config, config.get('fold_splitting_var', 'eventNumber'))
                for j, fold in enumerate(data_folds):
                    sub_dir_name = Path(file_names_in[i]).stem
                    file_path_out = os.path.join(config['output_dir'], sub_dir_name ,f"fold_{j}.root")
                    write_file(file_path_out, fold, tree_name=config.get('out_tree_name', 'tree'))
        else:
            for i, data in enumerate(data_out):
                sub_dir_name = Path(file_names_in[i]).stem
                file_path_out = os.path.join(config['output_dir'], sub_dir_name ,f"merged_total.root")
                write_file(file_path_out, data, tree_name=config.get('out_tree_name', 'tree'))

    print("Finished.")
if __name__ == "__main__":
    main()




