"""Run the six-jet fold-0 recipe from a checkout, saving its resolved configurations."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import uproot
import yaml


HERE = Path(__file__).resolve().parent


def read_yaml(path):
    return yaml.safe_load(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "dataset",
                        help="Directory containing the preprocessed ROOT folds")
    parser.add_argument("--output-dir", type=Path, help="Base directory for run outputs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--teacher-fraction", type=float, help="Override the training mixture (0 for truth only)")
    mode.add_argument("--test-checkpoint", type=Path, help="Score the held-out fold with this checkpoint")
    args = parser.parse_args()

    data = read_yaml(HERE / "data_config.yaml")
    model = read_yaml(HERE / "model_config.yaml")
    run = read_yaml(HERE / "run_fold_0.yaml")
    saved_model = None

    if args.test_checkpoint is None:
        stage = "fit"
        # Use Synapse's post-fit prediction loop to export the best epoch on validation.
        data["test_files"] = data["val_files"]
        run["run_mode"] = ["train", "test"]
        run["test_output"] = "validation_scores.root"
        if args.teacher_fraction is not None:
            model["loss_function"]["params"]["teacher_fraction"] = args.teacher_fraction
        needed = ("train_files", "val_files")
    else:
        stage = "test"
        checkpoint = args.test_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f"Missing checkpoint {checkpoint}")
        # The checkpoint directory sits beside the exact model config used for the fit.
        saved_model = checkpoint.parent.parent / "model_config.yaml"
        if saved_model.is_file():
            model = read_yaml(saved_model)
        run.update(run_mode=["test"], load_model=str(checkpoint), checkpoint_dir=None,
                   test_output="test_scores.root")
        needed = ("test_files",)

    fraction = model["loss_function"]["params"]["teacher_fraction"]
    if not 0 <= fraction <= 1:
        parser.error("teacher_fraction must be between 0 and 1")
    for key in ("train_files", "val_files", "test_files"):
        data[key] = [str((args.data_dir.expanduser() / f).resolve()) for f in data[key]]
        for path in data[key]:
            if key in needed and not Path(path).is_file():
                parser.error(f"Missing {path}. Set --data-dir to the directory containing the six-jet folds.")

    base = (args.output_dir or HERE / run["run_dir"]).expanduser().resolve()
    variant = f"evenet_{fraction:g}" if fraction else "truth_only"
    if saved_model is not None and not saved_model.is_file():
        variant = "heldout"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    dest = base / variant / f"{stage}-{timestamp}"
    dest.mkdir(parents=True)
    run["run_dir"] = str(dest)
    for name, cfg in (("data", data), ("model", model), ("run", run)):
        (dest / f"{name}_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Writing {stage} outputs to {dest}", flush=True)

    from synapse.main import main as run_synapse

    uproot.open.defaults["handler"] = uproot.source.file.MemmapSource
    run_synapse(["-d", str(dest / "data_config.yaml"), "-m", str(dest / "model_config.yaml"),
                 "-r", str(dest / "run_config.yaml")])


if __name__ == "__main__":
    main()
