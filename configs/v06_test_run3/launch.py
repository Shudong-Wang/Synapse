"""Run a six-jet fold recipe from a checkout, saving its resolved configurations."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import uproot
import yaml

from process_catalog import add_process_targets


HERE = Path(__file__).resolve().parent


def read_yaml(path):
    return yaml.safe_load(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=HERE,
                        help="Directory containing the data, model and selected fold's run configuration")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0, help="Held-out fold")
    parser.add_argument("--data-dir", type=Path, help="Preprocessed ROOT folds (default: CONFIG_DIR/dataset)")
    parser.add_argument("--output-dir", type=Path, help="Base directory for run outputs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--teacher-fraction", type=float, help="Override the training mixture (0 for truth only)")
    mode.add_argument("--test-checkpoint", type=Path, help="Score the held-out fold with this checkpoint")
    mode.add_argument("--extend-checkpoint", type=Path, help="Continue a completed fit from its last epoch checkpoint")
    parser.add_argument("--extra-epochs", type=int, help="Number of additional epochs for an extension")
    parser.add_argument("--restart-lr", type=float, help="Peak of the extension cosine learning-rate schedule")
    args = parser.parse_args()

    config_dir = args.config_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser() if args.data_dir is not None else config_dir / "dataset"
    data = read_yaml(config_dir / "data_config.yaml")
    model = read_yaml(config_dir / "model_config.yaml")
    run = read_yaml(config_dir / f"run_fold_{args.fold}.yaml")
    saved_model = None
    checkpoint_arg = args.test_checkpoint or args.extend_checkpoint
    if checkpoint_arg is not None:
        checkpoint = checkpoint_arg.expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f"Missing checkpoint {checkpoint}")
        saved_model = checkpoint.parent.parent / "model_config.yaml"
        if saved_model.is_file():
            model = read_yaml(saved_model)
    if args.extend_checkpoint:
        import torch

        if not args.extra_epochs or args.extra_epochs < 1 or not args.restart_lr or args.restart_lr <= 0:
            parser.error("Extending a fit requires positive --extra-epochs and --restart-lr")
        completed = torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"] + 1
        saved_run = checkpoint.parent.parent / "run_config.yaml"
        if saved_run.is_file():
            # Preserve the fit's batch size, seed and numerical settings at the new site.
            previous = read_yaml(saved_run)
            for key in ("batch_size", "num_workers", "seed", "use_amp", "val_sanity_check"):
                run[key] = previous[key]
        run.update(epochs=completed + args.extra_epochs, resume_checkpoint=str(checkpoint),
                   cosine_restart={"epoch": completed, "lr": args.restart_lr})
    elif args.extra_epochs is not None or args.restart_lr is not None:
        parser.error("--extra-epochs and --restart-lr apply to --extend-checkpoint")

    if args.test_checkpoint is None:
        stage = "extend" if args.extend_checkpoint else "fit"
        # Use Synapse's post-fit prediction loop to export the best epoch on validation.
        data["test_files"] = data["val_files"]
        run["run_mode"] = ["train", "test"]
        run["test_output"] = "validation_scores.root"
        if args.teacher_fraction is not None:
            if "teacher_fraction" not in model["loss_function"]["params"]:
                parser.error("--teacher-fraction applies to a mixed-target fit. This recipe has a separate EveNet head.")
            model["loss_function"]["params"]["teacher_fraction"] = args.teacher_fraction
        needed = ("train_files", "val_files")
    else:
        stage = "test"
        data["inputs"].pop("relation_info", None)
        run.update(run_mode=["test"], load_model=str(checkpoint), checkpoint_dir=None,
                   test_output="test_scores.root")
        needed = ("test_files",)

    params = model["loss_function"]["params"]
    if "teacher_fraction" in params:
        fraction = params["teacher_fraction"]
        if not 0 <= fraction <= 1:
            parser.error("teacher_fraction must be between 0 and 1")
        variant = f"evenet_{fraction:g}" if fraction else "truth_only"
    else:
        variant = "truth_with_aux"
        if stage != "test" and params.get("process_loss_weight", 0) > 0:
            outputs = model["model_params"]["event_aux_outputs"]["process"]
            add_process_targets(data, outputs, data_dir / "summary.json")
    for key in ("train_files", "val_files", "test_files"):
        data[key] = [str((data_dir / f).resolve()) for f in data[key]]
        for path in data[key]:
            if key in needed and not Path(path).is_file():
                parser.error(f"Missing {path}. Set --data-dir to the directory containing the six-jet folds.")

    base = (args.output_dir or config_dir / run["run_dir"]).expanduser().resolve()
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
