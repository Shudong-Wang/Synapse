"""Plot background yields, parent-process fractions and SM HH efficiency in the highest native Trafo60 bins."""

import csv
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
import yaml


COLORS = ("#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#9BA3A9")
PANEL_TITLES = {"Synapse": "Synapse · Synapse SR", "EveNet": "EveNet · EveNet SR",
                "XGB_SynapseSR": "XGBoost · Synapse SR", "XGB_EveNetSR": "XGBoost · EveNet SR",
                "Hybrid": "Physics-corrected EveNet · Hybrid SR"}


def bin_contents(data, groups, names, config, native):
    fold = config["event_number_remainder"]
    fold_key = f"fold_{fold}"
    if not np.all(data["eventNumber"] % 3 == fold):
        raise ValueError(f"The score tree and Trafo60 benchmark must both come from {fold_key}.")
    inputs, = config["input_data"]
    distribution = config["score_distribution"]
    fine = np.linspace(distribution["x_min"], distribution["x_max"], inputs["base_bin_num"] + 1)
    centers = (fine[:-1] + fine[1:]) / 2
    results = {}
    for method, classifier in inputs["classifiers"].items():
        reference = native[method][fold_key]
        edges = np.asarray(reference["test_bin_edges"], dtype=np.float64)
        if edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"{method} has invalid Trafo60 bin edges.")
        fields = {alias: data[column.format(model_name=inputs["model_name"])]
                  for alias, column in classifier["columns"].items()}
        frame = pd.DataFrame(fields, copy=False)
        score = frame.eval(classifier["expression"], engine="python").to_numpy(dtype=np.float64)
        selection = classifier["selection"]
        region = frame.eval(selection, engine="python").to_numpy(dtype=bool)
        codes = groups[region]
        score = score[region]
        weight = data[inputs["weight_col"]][region].astype(np.float64)
        # Reproduce hhml_eval's two histogram steps, including its fine-bin boundary convention.
        fine_yields = []
        for code in range(-1, len(names)):
            selected = codes == code
            yields, _ = np.histogram(score[selected], bins=fine, weights=weight[selected])
            fine_yields.append(yields)
        fine_yields = np.stack(fine_yields, axis=1)
        binned = np.array([fine_yields[(centers >= lo) & (centers <= hi)].sum(axis=0)
                           for lo, hi in zip(edges[:-1], edges[1:])])
        for calculated, key in [(binned[:, 0], "test_signal_hists"), (binned[:, 1:].sum(axis=1), "test_background_hists")]:
            if not np.allclose(calculated, reference[key], rtol=1e-7, atol=1e-9):
                raise ValueError(f"{method} {fold_key} {key} differs from the native benchmark. Check the score file and dataset.")
        signal_total = weight[codes == -1].sum()
        if signal_total <= 0:
            raise ValueError(f"The signed SM HH yield in the {method} signal region is {signal_total}.")
        bins = []
        for index in range(max(0, len(binned) - 3), len(binned)):
            signal = binned[index, 0]
            yields = binned[index, 1:]
            background = yields.sum()
            if background <= 0:
                raise ValueError(f"{method} bin {index + 1} needs positive total background to define its composition.")
            bins.append(dict(fold=fold, source_root=classifier.get("source_root", config["source_root"]), method=method,
                             score=classifier["expression"], selection=selection,
                             bin=index + 1, score_low=edges[index], score_high=edges[index + 1],
                             signal_sumw=signal, signal_region_sumw=signal_total, signal_eff=signal/signal_total,
                             background_sumw=background, yields=yields, fractions=yields/background))
        results[method] = bins
    return results


def plot_composition(panels, names, fold, out, titles=None):
    titles = PANEL_TITLES if titles is None else titles
    ncols = min(2, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig = plt.figure(figsize=(15, 2.65 * nrows + 1.8))
    grid = fig.add_gridspec(nrows, ncols, left=.104, right=.975, bottom=.12, top=.77, hspace=.94, wspace=.36)
    all_bins = [b for bins in panels.values() for b in bins]
    negative_limit = min(np.minimum(b["fractions"], 0).sum() for b in all_bins)
    positive_limit = max(np.maximum(b["fractions"], 0).sum() for b in all_bins)
    signal_limit = max(b["signal_eff"] for b in all_bins)
    for i, (method, bins) in enumerate(panels.items()):
        cell = grid[i // ncols, i % ncols].subgridspec(1, 3, width_ratios=(3.45, 1.05, 1.55), wspace=.13)
        comp = fig.add_subplot(cell[0])
        bkg = fig.add_subplot(cell[1], sharey=comp)
        sig = fig.add_subplot(cell[2], sharey=comp)
        y = np.arange(len(bins))
        fractions = np.stack([b["fractions"] for b in bins])
        left_positive, left_negative = np.zeros(len(bins)), np.zeros(len(bins))
        for j, color in enumerate(COLORS[:len(names)]):
            frac = fractions[:, j]
            left = np.where(frac >= 0, left_positive, left_negative)
            comp.barh(y, frac, left=left, color=color, height=.69, edgecolor="white", linewidth=.5)
            for row in np.flatnonzero(np.abs(frac) >= .145):
                comp.text(left[row] + frac[row]/2, row, f"{frac[row]:.0%}", ha="center", va="center",
                          fontsize=8, color="white" if j in (0, 2, 5) else "#172127")
            left_positive += np.maximum(frac, 0)
            left_negative += np.minimum(frac, 0)
        comp.set_xlim(min(0, negative_limit * 1.08), max(1, positive_limit * 1.025))
        comp.set_xticks([0, .25, .5, .75, 1])
        comp.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        labels = [f"{b['score_low']:.5f} – {b['score_high']:.5f}" for b in bins]
        if bins[-1]["score_high"] >= 1.0:
            labels[-1] = f"≥ {bins[-1]['score_low']:.5f}"
        comp.set_yticks(y, labels=labels)
        comp.tick_params(axis="y", length=0, pad=5, labelsize=8.6)
        comp.set_title(titles.get(method, method), loc="left", fontsize=13, fontweight="semibold", pad=10)
        comp.set_xlabel("Background composition", fontsize=10)
        totals = [b["background_sumw"] for b in bins]
        bkg.set_title("Bkg events", fontsize=10, pad=10)
        bkg.spines[:].set_visible(False)
        bkg.set_xticks([])
        bkg.yaxis.set_visible(False)
        for row, total in enumerate(totals):
            bkg.text(.5, row, f"{total:,.2f}", transform=bkg.get_yaxis_transform(),
                     ha="center", va="center", fontsize=10)
        bkg.set_xlabel(f"Σ {sum(totals):,.2f}", fontsize=10)
        efficiencies = np.array([b["signal_eff"] for b in bins])
        sig.barh(y, efficiencies, color="#384C5A", height=.49)
        for row, eff in enumerate(efficiencies):
            sig.annotate(f"{eff:.1%}", (eff, row), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5)
        sig.set_xlim(0, signal_limit * 1.55)
        sig.xaxis.set_major_locator(MaxNLocator(nbins=3))
        sig.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        sig.tick_params(axis="y", left=False, labelleft=False)
        sig.set_title("SM HH efficiency", fontsize=10, pad=10)
        sig.set_xlabel(f"Σ {efficiencies.sum():.1%}", fontsize=10)
        for ax in (comp, bkg, sig):
            ax.xaxis.set_label_coords(.5, -.30)
        for ax in (comp, sig):
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=8.5)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="0.91", lw=.6)
    title = "Background yield, composition and signal efficiency in the last three Trafo60 bins"
    if any(len(bins) < 3 for bins in panels.values()):
        title = "Background yield, composition and signal efficiency in the available high-score Trafo60 bins"
    fig.suptitle(title,
                 x=.035, y=.976, ha="left", fontsize=18, fontweight="semibold")
    fig.text(.035, .912, f"Fold {fold} · Background yields use signed nominal MC weights. "
             "SM HH efficiency is relative to the corresponding signal region. Σ sums the displayed bins.", fontsize=10)
    handles = [Patch(facecolor=color, label=name) for name, color in zip(names, COLORS)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.03, .887), ncol=len(names),
               frameon=False, fontsize=9.5, columnspacing=1.45)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"trafo60_last_three_bins.{suffix}", dpi=150)
    plt.close(fig)


def write_composition(data, groups, names, benchmark, out, titles=None):
    config = yaml.safe_load((benchmark / "benchmark.yaml").read_text())
    native = json.loads((benchmark / "sig_with_trafo.json").read_text())
    panels = bin_contents(data, groups, names, config, native)
    rows = []
    for bins in panels.values():
        for b in bins:
            values = {key: val for key, val in b.items() if key not in ("yields", "fractions")}
            for name, count, fraction in zip(names, b["yields"], b["fractions"]):
                rows.append(dict(**values, process=name, process_sumw=count, background_fraction=fraction))
    with (out / "trafo60_last_three_bins.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_composition(panels, names, config["event_number_remainder"], out, titles=titles)
