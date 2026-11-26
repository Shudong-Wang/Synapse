"""Compare event classifiers at matched SM HH efficiency on a scored Run 3 fold."""

import argparse
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, PercentFormatter
import numpy as np
import uproot
import yaml

from process_catalog import GROUPS, process_codes
from trafo60_composition import write_composition

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hhml_eval"))
from roc import weighted_roc_curve


TITLES = {"Vjets": "V + jets", "Vy": "Vγ", "top": "Top", "singleH": "Single Higgs"}
METHODS = {"Synapse": "p_signal", "EveNet": "evenet_pS", "XGBoost": "xgb_score"}
STYLES = {"Synapse": ("#20639B", "-"), "EveNet": ("#B94F19", "--"),
          "XGBoost": ("#237855", "-."), "Hybrid": ("#853F82", ":")}
MODES = {"within_sr": "Background rejection within each signal region",
         "from_preselection": "Background rejection including the conversion veto"}
TARGETS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
REFERENCE = "Synapse"


def default_catalog(scores):
    config = scores.with_name("data_config.yaml")
    if config.is_file():
        key = "val_files" if scores.name.startswith("validation") else "test_files"
        paths = yaml.safe_load(config.read_text()).get(key, [])
        directories = {(scores.parent / path).parent for path in paths}
        if len(directories) == 1:
            summary = directories.pop() / "summary.json"
            if summary.is_file():
                return summary
    raise ValueError("Pass --catalog with the preprocessing summary.json or a DSID map CSV.")


def load_scores(path, catalog):
    fields = ["eventNumber", "dsid", "class_id", "n_jet", "weight", "p_extconv_bkg",
              "evenet_valid", "evenet_pConv", *METHODS.values()]
    with uproot.open(path) as source:
        tree = source["tree"]
        hybrid = {"p_hybrid_signal", "p_hybrid_extconv_bkg"}
        available = hybrid.intersection(tree.keys())
        if available and available != hybrid:
            raise ValueError("The hybrid score needs both its HH and conversion probabilities.")
        data = tree.arrays([*fields, *sorted(available)], library="np")
    groups = process_codes(data["dsid"], catalog)
    if not np.isin(data["class_id"], (0, 1, 2)).all() or not np.array_equal(groups == -1, data["class_id"] == 0):
        raise ValueError("Scored MC classes and the parent-process catalogue disagree.")
    if not np.isfinite(data["weight"]).all():
        raise ValueError("Found a non-finite event weight.")
    return data, groups


def roc_inputs(data, groups):
    use = (data["n_jet"] >= 1) & (data["weight"] > 0)
    groups = groups[use]
    if not np.any(groups == -1) or not np.any(groups >= 0):
        raise ValueError("The jet preselection needs positive-weight signal and background events.")
    folds = np.unique(data["eventNumber"][use] % 3)
    fold_label = ("Fold " if len(folds) == 1 else "Folds ") + ", ".join(str(int(fold)) for fold in folds)
    weight = data["weight"][use].astype(np.float64)
    conv = data["p_extconv_bkg"][use]
    if not ((conv >= 0) & (conv <= 1)).all():
        raise ValueError("Invalid Synapse conversion probabilities in the preselection.")
    eve = data["evenet_valid"][use].astype(bool) & (data["evenet_pConv"][use] < 0.8)
    regions = {"Synapse": conv < 0.8, "EveNet": eve, "XGBoost": eve}
    scores = {method: data[field][use] for method, field in METHODS.items()}
    if "p_hybrid_signal" in data:
        conversion = data["p_hybrid_extconv_bkg"][use]
        if not ((conversion >= 0) & (conversion <= 1)).all():
            raise ValueError("Invalid hybrid conversion probabilities in the preselection.")
        regions["Hybrid"] = conversion < 0.8
        scores["Hybrid"] = data["p_hybrid_signal"][use]
    for method, score in scores.items():
        inside = score[regions[method]]
        if not ((inside >= 0) & (inside <= 1)).all():
            raise ValueError(f"Invalid {method} signal score in its signal region.")
    return groups, weight, regions, scores, fold_label


def signal_working_points(signal, weight, regions, scores):
    """Find inclusive score thresholds. Tied scores all pass the same cut."""
    total = weight[signal].sum()
    acceptances, cuts = {}, {}
    for method, score in scores.items():
        selected = signal & regions[method]
        if not selected.any():
            raise ValueError(f"No SM HH events in the {method} signal region.")
        selected_scores = score[selected]
        order = np.argsort(-selected_scores, kind="stable")
        ranked = selected_scores[order]
        neg_ranked = -ranked
        cumulative = np.cumsum(weight[selected][order])
        acceptances[method] = cumulative[-1] / total
        for mode in MODES:
            denominator = cumulative[-1] if mode == "within_sr" else total
            for target in TARGETS:
                index = np.searchsorted(cumulative, target * denominator)
                if index < len(ranked):
                    cut = ranked[index]
                    last = np.searchsorted(neg_ranked, -cut, side="right") - 1
                    cuts[method, mode, target] = float(cut), cumulative[last] / denominator
    return acceptances, cuts


def working_points(category, weight, regions, scores, cuts, reference=REFERENCE):
    """Propagate shared background MC fluctuations at fixed signal cuts."""
    rows = []
    total = weight.sum()
    sr_totals = {method: weight[region].sum() for method, region in regions.items()}
    for mode in MODES:
        for target in TARGETS:
            influences, values = {}, {}
            for method, region in regions.items():
                key = method, mode, target
                denom = sr_totals[method] if mode == "within_sr" else total
                if key not in cuts or denom == 0:
                    continue
                cut, achieved = cuts[key]
                passed = region & (scores[method] >= cut)
                kept = weight[passed].sum()
                rejection = denom / kept if kept > 0 else np.inf
                error = np.nan
                if kept > 0:
                    # Derivative of log(D/P) for each common, independently fluctuating event.
                    in_denom = region if mode == "within_sr" else 1.0
                    change = weight * (in_denom / denom - passed / kept)
                    influences[method] = change
                    error = rejection * np.linalg.norm(change)
                values[method] = dict(normalization=mode, category=category, method=method,
                                      target_signal_eff=target, signal_eff=achieved, score_min=cut,
                                      background_denominator_sumw=denom, background_passing_sumw=kept,
                                      positive_bkg_passing=np.count_nonzero(passed), background_eff=kept/denom,
                                      rejection=rejection, rejection_mc_stat=error)
            for method, point in values.items():
                ratio, error = np.nan, np.nan
                if method in influences and reference in influences:
                    ratio = point["rejection"] / values[reference]["rejection"]
                    error = ratio * np.linalg.norm(influences[method] - influences[reference])
                rows.append({**point, f"rejection_ratio_to_{reference.lower()}": ratio, "ratio_mc_stat": error})
    return rows


def measure(groups, weight, regions, scores, reference=REFERENCE):
    signal = groups == -1
    acceptances, cuts = signal_working_points(signal, weight, regions, scores)
    categories = {"All backgrounds": groups >= 0}
    categories.update({name: groups == i for i, name in enumerate(GROUPS.values()) if np.any(groups == i)})
    curves, points = {}, []
    for category, background in categories.items():
        bg_weight = weight[background]
        bg_regions = {m: mask[background] for m, mask in regions.items()}
        bg_scores = {m: score[background] for m, score in scores.items()}
        points.extend(working_points(category, bg_weight, bg_regions, bg_scores, cuts, reference))
        for method, region in regions.items():
            if not bg_regions[method].any():
                continue
            pick = region & (signal | background)
            fpr, tpr, _ = weighted_roc_curve(signal[pick], scores[method][pick], weight[pick])
            bg_acceptance = bg_weight[bg_regions[method]].sum() / bg_weight.sum()
            curves["within_sr", category, method] = tpr, fpr
            curves["from_preselection", category, method] = tpr * acceptances[method], fpr * bg_acceptance
    return list(categories), curves, points


def matched_curve(curves, mode, category, method, reference_method=REFERENCE):
    x, eff = curves[mode, category, method]
    reference = curves.get((mode, category, reference_method))
    ratio = np.full_like(x, np.nan)
    if reference is not None:
        ref_x, ref_eff = reference
        # Include the exact end of the common acceptance, even on unequal grids.
        if 0 < ref_x[-1] < x[-1]:
            grid = np.union1d(x, ref_x[-1])
            eff, x = np.interp(grid, x, eff), grid
        ref = np.interp(x, ref_x, ref_eff, left=np.nan, right=np.nan)
        ratio = np.divide(ref, eff, out=np.full_like(x, np.nan), where=(eff > 0) & (ref > 0))
    rejection = np.divide(1.0, eff, out=np.full_like(x, np.inf), where=eff > 0)
    return x, eff, rejection, ratio


def ratio_axis(axis, chunks):
    values = np.concatenate(chunks)
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = values.min(), values.max()
    if lo >= 0.42 and hi <= 2.15:
        lo, hi, ticks = 0.42, 2.15, [0.5, 1, 2]
    else:
        pad = (hi / lo) ** 0.04
        lo, hi = min(lo / pad, 0.75), max(hi * pad, 1.3)
        decades = 10.0 ** np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
        ticks = np.outer(decades, [1] if hi / lo > 55 else [1, 2, 5]).ravel()
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
    axis.set_yticks(ticks, labels=[f"{v:g}" for v in ticks])
    axis.set_ylim(lo, hi)
    axis.yaxis.set_minor_locator(NullLocator())


def plot(categories, curves, points, fold_label, out, styles=None, labels=None, reference=REFERENCE):
    methods = {method for _, _, method in curves}
    styles = STYLES if styles is None else styles
    labels = {} if labels is None else labels
    ratio_key = f"rejection_ratio_to_{reference.lower()}"
    handles = [Line2D([], [], color=c, ls=ls, lw=2, label=labels.get(m, m)) for m, (c, ls) in styles.items() if m in methods]
    notes = {"within_sr": "Efficiencies use each method’s events after its conversion veto.",
             "from_preselection": "Efficiencies use the same jet-preselected events before the conversion veto."}
    with PdfPages(out / "rejection_by_background.pdf") as pdf, (out / "roc_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["normalization", "category", "method", "signal_eff", "background_eff",
                         "rejection", ratio_key])
        for mode, title in MODES.items():
            fig = plt.figure(figsize=(14.5, 9.5))
            grid = fig.add_gridspec(2, 4, left=.072, right=.985, bottom=.084, top=.855, wspace=.29, hspace=.35)
            for i, category in enumerate(categories):
                panel = grid[i // 4, i % 4].subgridspec(2, 1, height_ratios=(2.15, 1), hspace=.075)
                main = fig.add_subplot(panel[0])
                lower = fig.add_subplot(panel[1], sharex=main)
                for ax in main, lower:
                    ax.set_yscale("log")
                    ax.grid(axis="y", which="major", color="0.88", lw=.7)
                    ax.spines[["right", "top"]].set_visible(False)
                    ax.tick_params(labelsize=9)
                lower.axhline(1, color="0.65", lw=.7)
                visible = [np.array([1.0])]
                for method, (color, style) in styles.items():
                    if (mode, category, method) not in curves:
                        continue
                    x, eff, rejection, ratio = matched_curve(curves, mode, category, method, reference)
                    writer.writerows((mode, category, method, *row) for row in zip(x, eff, rejection, ratio))
                    main.plot(x, np.where((x >= .1) & np.isfinite(rejection), rejection, np.nan),
                              color=color, ls=style, lw=1.65)
                    lower.plot(x, np.where(x >= .1, ratio, np.nan), color=color, ls=style, lw=1.45)
                    visible.append(ratio[x >= .1])
                    markers = [p for p in points if (p["normalization"], p["category"], p["method"]) ==
                               (mode, category, method) and p["target_signal_eff"] in (.2, .5, .7)]
                    for ax, key, err in [(main, "rejection", "rejection_mc_stat"),
                                         (lower, ratio_key, "ratio_mc_stat")]:
                        if ax is lower and method == reference:
                            continue
                        pts = [p for p in markers if np.isfinite(p[key]) and np.isfinite(p[err])]
                        if pts:
                            y = np.array([p[key] for p in pts])
                            dy = np.array([p[err] for p in pts])
                            ax.errorbar([p["signal_eff"] for p in pts], y, yerr=dy, fmt="none",
                                        ecolor=color, elinewidth=1, capsize=2.3)
                            if ax is lower:
                                visible.extend((y - dy, y + dy))
                ratio_axis(lower, visible)
                main.set_title(TITLES.get(category, category), loc="left", fontsize=13, fontweight="semibold")
                main.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
                lower.set_xlim(.1, 1)
                lower.set_xticks([.2, .4, .6, .8, 1])
                lower.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                if i % 4 == 0:
                    main.set_ylabel("Background rejection", fontsize=11)
                    lower.set_ylabel(f"Ratio to\n{labels.get(reference, reference)}", fontsize=10)
            fig.suptitle(title, x=.067, y=.98, ha="left", fontsize=20, fontweight="semibold")
            if len(handles) > 4:
                fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.061, .957), ncol=len(handles),
                           frameon=False, fontsize=10, columnspacing=1.45, borderaxespad=.1)
            else:
                fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.99, .977), ncol=len(handles),
                           frameon=False, fontsize=11, columnspacing=1.1)
            fig.text(.067, .922, notes[mode] + "  Lower panels compare rejection at matched SM HH efficiency.", fontsize=10.5)
            fig.text(.067, .893, f"{fold_label} · Positive MC weights · Whiskers at 20%, 50% and 70% show background MC statistical uncertainty.",
                     fontsize=9.5, color="0.30")
            fig.supxlabel("SM HH signal efficiency", y=.018, fontsize=13)
            pdf.savefig(fig)
            fig.savefig(out / f"rejection_{mode}.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="Synapse validation_scores.root or test_scores.root")
    parser.add_argument("--catalog", type=Path, help="Per-part preprocessing summary.json or CSV with dsid,analysis_category")
    parser.add_argument("--benchmark", type=Path, help="Matching hhml_eval fold directory containing benchmark.yaml and sig_with_trafo.json")
    parser.add_argument("--out", type=Path, help="Output directory (default: background_rejection next to the scores)")
    args = parser.parse_args()
    catalog = args.catalog or default_catalog(args.scores)
    data, groups = load_scores(args.scores, catalog)
    roc_groups, weight, regions, scores, fold_label = roc_inputs(data, groups)
    categories, curves, points = measure(roc_groups, weight, regions, scores)
    if not curves:
        raise ValueError("No signal region contains a positive-weight background for a rejection curve.")
    out = args.out or args.scores.parent / "background_rejection"
    role = "validation" if args.scores.name.startswith("validation") else "held_out"
    benchmark = args.benchmark
    if benchmark is None:
        workflow = args.scores.resolve().parent.parent.parent
        for base in (workflow / "evaluation", out.parent):
            candidate = base / "hhml_eval" / role
            if (candidate / "sig_with_trafo.json").is_file():
                benchmark = candidate
                break
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".plots-", dir=out) as tmp:
        stage = Path(tmp)
        with (stage / "working_points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
        plot(categories, curves, points, fold_label, stage)
        if benchmark is not None:
            names = [TITLES.get(name, name) for name in GROUPS.values()]
            write_composition(data, groups, names, benchmark, stage)
        else:
            print("To include the final three Trafo60 bins, supply the matching hhml_eval fold directory with --benchmark.")
        for path in stage.iterdir():
            path.replace(out / path.name)
    print(f"{fold_label} · {len(weight):,} positive-weight jet-preselected MC events · {out / 'rejection_by_background.pdf'}")


if __name__ == "__main__":
    main()
