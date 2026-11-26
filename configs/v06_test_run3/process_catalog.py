"""Run 3 generator-process labels shared by fitting and background-rejection plots."""

import csv
import json
from collections import defaultdict

import numpy as np

from preprocess import SM_HH


GROUPS = {"VV__WZ": "WZ", "VV__otherVV": "Other VV", "VVV": "VVV",
          "Vjets": "Vjets", "Vy": "Vy", "top": "top", "singleH": "singleH"}
PROCESS_OUTPUTS = {"hhml": "p_process_hh", "WZ": "p_process_wz", "Other VV": "p_process_other_vv",
                   "VVV": "p_process_vvv", "Vjets": "p_process_vjets", "Vy": "p_process_vgamma",
                   "top": "p_process_top", "singleH": "p_process_single_h"}


def parent_processes(catalog):
    parents = defaultdict(set)
    if catalog.suffix.lower() == ".csv":
        with catalog.open(newline="") as stream:
            for row in csv.DictReader(stream):
                parents[int(row["dsid"])].add(row["analysis_category"])
    else:
        manifest = json.loads(catalog.read_text())
        if "parts" not in manifest:
            raise ValueError(f"{catalog} has no per-part process labels. Supply a DSID map CSV.")
        for part in manifest["parts"]:
            dsid = int(part["part"].split("_", 1)[0])
            for process, count in part["kept"].items():
                if count:
                    parents[dsid].add("hhml" if process in SM_HH else GROUPS.get(process, process))
    return parents


def process_code_map(parents, dsids, order):
    categories = {name: index for index, name in enumerate(order)}
    codes = {}
    for dsid in dsids:
        names = parents.get(dsid, set())
        if len(names) != 1 or not names <= categories.keys():
            raise ValueError(f"Expected one known parent process for DSID {dsid}, found {sorted(names)}")
        codes[int(dsid)] = categories[next(iter(names))]
    return codes


def process_codes(dsids, catalog):
    ids, inverse = np.unique(dsids, return_inverse=True)
    codes = process_code_map(parent_processes(catalog), ids, ("hhml", *GROUPS.values()))
    # The rejection plots use -1 for HH and zero-based background indices.
    return np.array([codes[i] - 1 for i in ids], dtype=np.int8)[inverse]


def add_process_targets(data, outputs, catalog):
    if list(outputs) != list(PROCESS_OUTPUTS.values()):
        raise ValueError(f"Expected the Run 3 process outputs in this order: {list(PROCESS_OUTPUTS.values())}")
    parents = parent_processes(catalog)
    codes = process_code_map(parents, sorted(parents), PROCESS_OUTPUTS)
    # Freeze the complete DSID lookup in the resolved data YAML so a fit is reproducible.
    data["new_variables"]["process_id"] = f"lookup_ids(dsid, {codes})"
    data["inputs"]["process_info"] = ["process_id", "abs_weight"]
