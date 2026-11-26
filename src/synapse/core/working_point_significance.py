"""Expected significance in disjoint score bins bounded by fixed HH efficiencies."""
import numpy as np


EFFICIENCIES = (0.20, 0.30, 0.50)


def asimov_significance(signal, background, variance):
    """Cowan Eq. (20) with a background MC variance, evaluated using log1p."""
    if signal <= 0 or background <= 0:
        return 0.0
    total = signal + background
    if variance == 0:
        half_q = total * np.log1p(signal / background) - signal
    else:
        first = total * np.log1p(signal * background / (background ** 2 + total * variance))
        second = background ** 2 / variance * np.log1p(
            variance * signal / (background * (background + variance)))
        half_q = first - second
    return float(np.sqrt(2 * max(0., half_q)))


def working_point_significance(score, is_hh, weight, preselection, signal_region, cuts=None):
    """Fit positive-weight HH cuts or apply supplied cuts and count signed yields and sum(w²).

    The efficiency denominator includes the full jet preselection before the conversion veto.
    The checkpoint metric combines the 0–20%, 20–30% and 30–50% bands in quadrature.
    Return None if the signal region cannot reach all three requested efficiencies when fitting.
    """
    score = np.asarray(score)
    hh = np.asarray(is_hh, dtype=bool)
    w = np.asarray(weight, dtype=np.float64)
    pre = np.asarray(preselection, dtype=bool)
    sr = pre & np.asarray(signal_region, dtype=bool)
    if not (score.ndim == 1 and score.shape == hh.shape == w.shape == pre.shape == sr.shape):
        raise ValueError("Expected one score, class, weight and pair of selection flags per event")
    if not np.isfinite(w[pre]).all() or not np.isfinite(score[sr]).all():
        raise ValueError("Nonfinite weight or score in the evaluated sample")
    positive_hh = hh & (w > 0)
    total_positive_hh = w[positive_hh & pre].sum()
    if total_positive_hh <= 0:
        return None
    if cuts is None:
        rows = np.flatnonzero(positive_hh & sr)
        order = rows[np.argsort(-score[rows], kind="stable")]
        indices = np.searchsorted(w[order].cumsum(), total_positive_hh * np.asarray(EFFICIENCIES))
        if indices[-1] >= len(order):
            return None
        cuts = score[order[indices]]
    cuts = np.asarray(cuts, dtype=np.float64)
    if cuts.shape != (len(EFFICIENCIES),) or not np.isfinite(cuts).all() or np.any(np.diff(cuts) > 0):
        raise ValueError("Expected three finite, descending inclusive cut values")

    # A tied score enters the tightest selected bin; equal cuts give an empty exclusive band.
    bins = np.searchsorted(-cuts, -score[sr], side="left")
    hh, w = hh[sr], w[sr]
    def yields(select, weights):
        return np.bincount(bins[select], weights=weights[select], minlength=len(cuts)+1)[:len(cuts)]

    s = yields(hh, w)
    b = yields(~hh, w)
    v = yields(~hh, w*w)
    achieved = yields(hh & (w > 0), w) / total_positive_hh
    def records(signal, background, variance, efficiency):
        return [dict(signal=float(si), background=float(bi), background_mc_variance=float(vi),
                     positive_hh_efficiency=float(e), z=asimov_significance(si, bi, vi))
                for si, bi, vi, e in zip(signal, background, variance, efficiency)]

    exclusive = records(s, b, v, achieved)
    cumulative = records(s.cumsum(), b.cumsum(), v.cumsum(), achieved.cumsum())
    combined = float(np.linalg.norm([band["z"] for band in exclusive]))
    return dict(efficiencies=list(EFFICIENCIES), cuts=cuts.tolist(), cumulative=cumulative,
                exclusive=exclusive, quadrature=combined, exclusive_quadrature=combined)


def hh_significance_20_30_50(logits, true, weight, selection_info):
    """Disjoint-bin checkpoint metric using jet count and signed nominal weight from selection_info."""
    p = logits.detach().softmax(dim=1).cpu().numpy()
    jets, signed = selection_info.detach().cpu().numpy().T
    result = working_point_significance(p[:, 0], true.detach().cpu().numpy() == 0, signed,
                                        jets >= 1, p[:, 2] < .8)
    return result["exclusive_quadrature"] if result is not None else 0.0
