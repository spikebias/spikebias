"""Spike assignment biases of sorted units.

Each sorted unit is compared with every ground truth neuron and labelled:

* false positive: none of its agreement scores is above chance;
* good: it matches a ground truth neuron with a score of at least 0.8;
* overmerger: it matches more than one ground truth neuron above chance;
* oversplitter: a ground truth neuron it matches is also matched above chance by another sorted unit.

A unit can be good, an overmerger and an oversplitter at once ("mixed" classes). The chance score of a
pair assumes two independent Poisson spike trains with the units' firing rates.
"""

import numpy as np
import pandas as pd

from .accuracy import DELTA_TIME, WELL_SORTED, agreement_scores
from .utils import get_duration, single_units, spike_counts, standardize_layers

BIAS_CLASSES = [
    "good",
    "overmerger",
    "oversplitter",
    "mixed: good + overmerger",
    "mixed: good + oversplitter",
    "mixed: overmerger + oversplitter",
    "mixed: good + overmerger + oversplitter",
    "false positive",
]


def chance_scores(
    firing_rates_sorted, firing_rates_true, duration: float, delta_time: float = DELTA_TIME
) -> np.ndarray:
    """Agreement scores expected by chance for all pairs of sorted units (rows) and ground truth neurons.

    Args:
        firing_rates_sorted: firing rates of the sorted units (spikes/s)
        firing_rates_true: firing rates of the ground truth neurons (spikes/s)
        duration: duration of the recording (s)
        delta_time: coincidence window of a hit (ms)
    """
    fr_s = np.asarray(firing_rates_sorted, dtype=float)[:, None]
    fr_g = np.asarray(firing_rates_true, dtype=float)[None, :]

    # probability that a spike of the sparser train has a chance coincidence
    p_hit = 1.0 - np.exp(-2 * delta_time * np.minimum(fr_g, fr_s) / 1000)

    # expected hits; the sparser spike train bounds the number of hits
    n_s, n_g = fr_s * duration, fr_g * duration
    n_hits = p_hit * np.minimum(n_g, n_s)
    with np.errstate(invalid="ignore", divide="ignore"):
        chance = n_hits / (n_s + n_g - n_hits)
    return np.nan_to_num(chance, nan=0.0)


def label_units(scores: np.ndarray, chances: np.ndarray, good_threshold: float = WELL_SORTED) -> np.ndarray:
    """Bias class of each sorted unit from its agreement and chance scores (sorted units x ground truth)."""
    scores, chances = np.asarray(scores, dtype=float), np.asarray(chances, dtype=float)
    matched = scores > 0
    above = scores > chances

    false_positive = ~(matched & above).any(axis=1)
    good = (matched & (scores >= good_threshold)).any(axis=1)
    overmerger = (matched & above).sum(axis=1) > 1

    # other sorted units that detect the same ground truth neurons above chance
    n_above = above.sum(axis=0)[None, :] - above
    oversplitter = (matched & (n_above > 0)).any(axis=1)

    labels = []
    for fp, g, om, os_ in zip(false_positive, good, overmerger, oversplitter):
        if fp:
            labels.append("false positive")
            continue
        parts = [name for name, flag in (("good", g), ("overmerger", om), ("oversplitter", os_)) if flag]
        if not parts:
            labels.append("unclassified")
        elif len(parts) == 1:
            labels.append(parts[0])
        else:
            labels.append("mixed: " + " + ".join(parts))
    return np.array(labels, dtype=object)


def classify_sorted_units(
    ground_truth,
    sorting,
    duration: float = None,
    delta_time: float = DELTA_TIME,
    good_threshold: float = WELL_SORTED,
    single_units_only: bool = True,
) -> pd.DataFrame:
    """Spike assignment bias of each sorted unit (see the module's description).

    Args:
        ground_truth: SortingExtractor of the ground truth neurons
        sorting: SortingExtractor of the spike sorter, over the same period
        duration: duration of the period (s); inferred from the extractors when possible
        delta_time: coincidence window of a hit (ms)
        good_threshold: agreement score of a "good" unit
        single_units_only: classify only the sorter's single-units (Kilosort "good" units)

    Returns:
        pd.DataFrame: indexed by sorted unit id, with the columns "bias" and, when the sorting
        has a "layer" property, "layer"
    """
    if single_units_only:
        sorting = single_units(sorting)
    duration = float(duration) if duration is not None else _duration(ground_truth, sorting)
    scores = agreement_scores(ground_truth, sorting, delta_time).T
    chances = chance_scores(
        spike_counts(sorting) / duration, spike_counts(ground_truth) / duration, duration, delta_time
    )
    df = pd.DataFrame({"bias": label_units(scores.values, chances, good_threshold)}, index=scores.index)
    if "layer" in sorting.get_property_keys():
        df["layer"] = standardize_layers(sorting.get_property("layer"))
    return df.rename_axis("unit_id")


def bias_proportions(biases: pd.DataFrame, by=None) -> pd.DataFrame:
    """Proportion of each bias class, overall or per group (e.g. by="layer")."""
    classes = BIAS_CLASSES + sorted(set(biases["bias"]) - set(BIAS_CLASSES))
    if by is None:
        counts = biases["bias"].value_counts()
        return (counts / counts.sum()).reindex(classes, fill_value=0.0).rename("proportion").to_frame()
    counts = biases.groupby(by)["bias"].value_counts().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).reindex(columns=classes, fill_value=0.0)


def _duration(ground_truth, sorting) -> float:
    """Duration of the period shared by the ground truth and the sorting."""
    for extractor in (ground_truth, sorting):
        try:
            return get_duration(extractor)
        except ValueError:
            continue
    raise ValueError("cannot infer the duration: pass duration (in seconds)")
