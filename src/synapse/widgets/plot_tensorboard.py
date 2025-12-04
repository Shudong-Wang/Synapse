#!/usr/bin/env python3
"""
Plot TensorBoard scalars grouped as train/val/test for common metrics.

Expected tag patterns (scalar names), e.g.:
    train_loss_step, val_loss_step, test_loss_step
    train_accuracy, val_accuracy, test_accuracy
    train_xxx, val_xxx, test_xxx

For each "xxx", this script will plot the available splits
(train / val / test) on the same figure.

Usage:
    python plot_tb_train_val_test.py /path/to/logdir_or_eventfile \
        --out_dir plots --show

Requirements:
    pip install tensorboard matplotlib
"""

import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

PHASES = ("train", "val", "test")


def load_scalars(log_path):
    """
    Load all scalar summaries from a TensorBoard log directory or event file.

    Returns
    -------
    dict
        {tag: {"steps": [...], "values": [...]} }
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Path does not exist: {log_path}")

    print(f"Loading scalars from: {log_path}")

    ea = event_accumulator.EventAccumulator(
        log_path,
        size_guidance={event_accumulator.SCALARS: 0},  # load all scalar points
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    scalars = {}

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalars[tag] = {"steps": steps, "values": values}

    return scalars


def group_train_val_test(scalars):
    """
    Group tags of the form '{phase}_{metric}' into a nested dict.

    Input:
        scalars: {tag: {"steps": [...], "values": [...]}}
    Output:
        metrics: {metric_name: {phase: {"steps": [...], "values": [...]}}}

    Example:
        'train_loss_step' -> phase='train', metric='loss_step'
        metrics['loss_step']['train'] = data
    """
    metrics = {}

    for tag, data in scalars.items():
        for phase in PHASES:
            prefix = phase + "_"
            if tag.startswith(prefix):
                metric = tag[len(prefix):]
                if metric == "":
                    # weird edge case, skip
                    continue
                metric = metric.replace("_step_step", "_step")  # clean up double _step
                metric = metric.replace("_step_epoch", "_epoch")  # clean up double _epoch
                if metric not in metrics:
                    metrics[metric] = {}
                metrics[metric][phase] = data
                break  # do not try other phases once matched

    return metrics


def plot_metrics(metrics, out_dir=None, show=False):
    """
    Plot all metrics. For each metric, plot the available phases (train/val/test)
    on the same figure.
    """
    if not metrics:
        print("No train_ / val_ / test_ scalar tags found.")
        return

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    print("Found metrics (common keys after 'train_/val_/test_'):")
    for m, phases in metrics.items():
        available = ", ".join(sorted(phases.keys()))
        print(f"  - {m}  (phases: {available})")

    for metric, phase_dict in metrics.items():
        plt.figure()

        # Plot phases in a consistent order
        for phase in PHASES:
            if phase not in phase_dict:
                continue  # skip missing val/test/etc.

            data = phase_dict[phase]
            steps = data["steps"]
            values = data["values"]
            markersize = 4 if len(steps) < 50 else 0
            if metric == "event_loss_epoch":
            # if metric == "loss_epoch":
                print(f"{phase} loss of the last epoch: {values[-1]}")
            plt.plot(steps, values, label=phase, marker="o", markersize=markersize, linewidth=1, alpha=0.8)

        plt.xlabel("Step")
        plt.ylabel(metric)
        # plt.yscale("log")
        plt.title(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if out_dir is not None:
            safe_metric = metric.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(out_dir, f"{safe_metric}.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=200)
            print(f"Saved plot: {out_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot train/val/test scalars with common keys from a TensorBoard log."
    )
    parser.add_argument(
        "log",
        help="Path to TensorBoard log directory or event file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively.",
    )

    args = parser.parse_args()

    scalars = load_scalars(args.log)
    metrics = group_train_val_test(scalars)

    # out_dir = args.out_dir if args.out_dir.strip() != "" else None
    out_dir = os.path.join(os.path.dirname(args.log + "/"), "train_val_test_plots")
    plot_metrics(metrics, out_dir=out_dir, show=args.show)


if __name__ == "__main__":
    main()
