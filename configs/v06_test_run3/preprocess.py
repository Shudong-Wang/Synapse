"""Build the three Synapse folds with up to six jets from the v06 SR and ExtConvCR ntuples."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot
import yaml


RUN = Path(__file__).resolve().parent
SM_HH = ("hhml__ggF__kl1__ALL__PhPy8EG", "hhml__VBF__kl1k2v1kv1__ALL__MGPy8EG")
BACKGROUNDS = ("VVV", "VV__WZ", "VV__otherVV", "Vjets", "Vy", "singleH", "top")
CLASSES = ("signal", "bkg", "extconv_bkg")
EVENET = ("evenet_pS", "evenet_pB", "evenet_pConv")
LEPTONS = ("l1", "l2", "l3")
JETS = ("j1", "j2", "j3", "j4", "j5", "j6")


def object_branches(field, jets):
    # The _nan branches have signed coordinates and mark missing jets consistently.
    leading = [f"{obj}_{field}" for obj in (*LEPTONS, jets[0])]
    return leading + [f"{jet}_{field}_nan" for jet in jets[1:]]


def convert(a, signal, event_features, region, object_fields):
    out = {key: a[key] for key in (*event_features, "eventNumber", "dsid", "weight",
                                   "is_conv", "mc_category", *EVENET)}
    out["xgb_score"] = a["XGB"]
    out["evenet_valid"] = np.logical_and.reduce([(a[key] >= 0) & (a[key] <= 1) for key in EVENET])
    out["class_id"] = np.where(signal, 0, 1 + a["is_conv"]).astype(np.int8)
    out["in_extconvcr"] = np.full(len(signal), region == "extconvcr")
    for i, name in enumerate(CLASSES):
        out[f"label_{name}"] = out["class_id"] == i

    zero = np.zeros_like(a["met_met"])
    met = {"pt": a["met_met"], "eta": zero, "phi": a["met_phi"], "e": a["met_met"]}
    for field, met_value in met.items():
        out[field] = np.column_stack([*(a[key] for key in object_fields[field]), met_value])
    valid = (out["pt"] > 0) & (out["e"] > 0)
    for field in met:
        out[field][~valid] = 0
    out["valid"] = valid
    for field in ("pdg", "charge"):
        out[field] = np.zeros(valid.shape, dtype=np.int8)
        out[field][:, :len(LEPTONS)] = np.column_stack([a[f"{obj}_{field}"] for obj in LEPTONS])
    out["isMET"] = np.zeros_like(valid)
    out["isMET"][:, -1] = valid[:, -1]
    out["isJet"] = np.zeros_like(valid)
    out["isJet"][:, len(LEPTONS):-1] = valid[:, len(LEPTONS):-1]
    for key in (*event_features, *met, "weight"):
        if not np.isfinite(out[key]).all():
            raise ValueError(f"Non-finite {key} in {region}")
    return out


def prepare(input_dir, destination, max_jets=len(JETS)):
    if not 1 <= max_jets <= len(JETS):
        raise ValueError(f"Expected between 1 and {len(JETS)} jet slots")
    jets = JETS[:max_jets]
    object_fields = {field: object_branches(field, jets) for field in ("pt", "eta", "phi", "e")}
    config = yaml.safe_load((RUN / "data_config.yaml").read_text())
    evt = config["inputs"]["evt_feats"]
    branches = set(evt) | set(EVENET) | {"sample_process", "dsid", "eventNumber", "weight", "is_conv",
                           "mc_category", "met_met", "met_phi", "XGB"}
    for names in object_fields.values():
        branches.update(names)
    branches.update(f"{obj}_{field}" for obj in LEPTONS for field in ("pdg", "charge"))

    counts = np.zeros((3, 3), dtype=np.int64)
    jet_counts = np.zeros((3, max_jets + 1), dtype=np.int64)
    sumabs = np.zeros((3, 3), dtype=np.float64)
    sources = {}
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Use --output for a new build.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".preprocess-") as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        with ExitStack() as stack:
            files = [stack.enter_context(uproot.recreate(stage / f"3fold_{i}.root",
                                                        compression=uproot.LZ4(4))) for i in range(3)]
            trees = []
            for region in ("sr", "extconvcr"):
                path = input_dir / f"{region}.root"
                region_counts = np.zeros_like(counts)
                invalid_evenet = np.zeros(3, dtype=np.int64)
                excluded = Counter()
                with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
                    tree = source[region]
                    for a in tree.iterate(sorted(branches), step_size=100_000, library="np"):
                        process = a.pop("sample_process")
                        signal = np.isin(process, SM_HH)
                        keep = signal | np.isin(process, BACKGROUNDS)
                        if not keep.all():
                            excluded.update(process[~keep].tolist())
                            if not keep.any():
                                continue
                            a = {k: v[keep] for k, v in a.items()}
                            signal = signal[keep]
                        if not np.array_equal(a["is_conv"], a["mc_category"] == 4):
                            raise ValueError(f"Inconsistent conversion truth in {path}")
                        out = convert(a, signal, evt, region, object_fields)
                        if not trees:
                            schema = {k: np.dtype((v.dtype, v.shape[1:])) for k, v in out.items()}
                            trees = [file.mktree("tree", schema) for file in files]
                        remainder = out["eventNumber"] % 3
                        for i, target in enumerate(trees):
                            pick = remainder == i
                            if not pick.any():
                                continue
                            target.extend({k: v[pick] for k, v in out.items()})
                            labels = out["class_id"][pick]
                            invalid_evenet[i] += np.count_nonzero(~out["evenet_valid"][pick])
                            region_counts[i] += np.bincount(labels, minlength=3)
                            jet_counts[i] += np.bincount(out["isJet"][pick].sum(axis=1), minlength=max_jets + 1)
                            sumabs[i] += np.bincount(labels, weights=np.abs(out["weight"][pick]), minlength=3)
                    entries = tree.num_entries
                if region_counts.sum() + sum(excluded.values()) != entries:
                    raise ValueError(f"Entry count mismatch in {path}")
                sources[region] = dict(file=str(path), entries=entries, excluded=dict(excluded),
                                       invalid_evenet_by_remainder=invalid_evenet.tolist(),
                                       counts_by_remainder=region_counts.tolist())
                counts += region_counts
                print(f"{region}: {region_counts.sum():,} retained of {entries:,} events", flush=True)

        if np.any(counts == 0) or np.any(sumabs <= 0):
            raise ValueError("Each fold must contain all three classes with positive total absolute weight")
        factors = sumabs[2, 1] / sumabs[2]
        if not np.allclose(factors, config["weights"]["weight_balance_factors"], rtol=1e-7, atol=0):
            raise ValueError(f"Update the fold-0 weight_balance_factors in data_config.yaml to {factors.tolist()}")
        summary = dict(classes=CLASSES, fold_0={"train": 2, "validation": 1, "test": 0},
                       object_slots=[*LEPTONS, *jets, "met"], jet_multiplicity_by_remainder=jet_counts.tolist(),
                       teacher_scores=EVENET, teacher_validity="evenet_valid",
                       sources=sources, counts_by_remainder=counts.tolist(),
                       sum_abs_weights_by_remainder=sumabs.tolist(), weight_balance_factors=factors.tolist())
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing sr.root and extconvcr.root")
    parser.add_argument("--output", type=Path, default=RUN / "dataset")
    parser.add_argument("--max-jets", type=int, choices=range(1, len(JETS) + 1), default=len(JETS),
                        help="Number of leading jet slots to store (use 2 for the earlier fold layout)")
    args = parser.parse_args()
    prepare(args.input_dir.expanduser().resolve(), args.output.expanduser().resolve(), args.max_jets)
