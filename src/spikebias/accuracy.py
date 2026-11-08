"""Sorting accuracy of ground truth neurons.

A sorted spike is a hit when it falls within delta_time (1.3 ms) of a ground truth spike. The agreement
score of a ground truth neuron and a sorted unit is hits / (hits + misses + false positives), and the
sorting accuracy of a ground truth neuron is its best agreement score with a sorted unit.
"""

import pandas as pd
from spikeinterface import comparison

from .utils import single_units

# coincidence window of a hit (ms)
DELTA_TIME = 1.3

# accuracy of a well-sorted neuron
WELL_SORTED = 0.8


def agreement_scores(ground_truth, sorting, delta_time: float = DELTA_TIME) -> pd.DataFrame:
    """Agreement scores of all pairs of ground truth neurons (rows) and sorted units (columns)."""
    comp = comparison.compare_sorter_to_ground_truth(
        ground_truth, sorting, exhaustive_gt=True, delta_time=delta_time
    )
    return comp.agreement_scores


def sorting_accuracy(
    ground_truth, sorting, delta_time: float = DELTA_TIME, single_units_only: bool = True
) -> pd.Series:
    """Sorting accuracy of each ground truth neuron: its best agreement score with a sorted unit.

    Args:
        ground_truth: SortingExtractor of the ground truth neurons
        sorting: SortingExtractor of the spike sorter
        delta_time: coincidence window of a hit (ms)
        single_units_only: match only the sorter's single-units (Kilosort "good" units)

    Returns:
        pd.Series: accuracy indexed by ground truth unit id
    """
    if single_units_only:
        sorting = single_units(sorting)
    scores = agreement_scores(ground_truth, sorting, delta_time)
    return scores.max(axis=1).rename("accuracy")


def best_matches(
    ground_truth, sorting, delta_time: float = DELTA_TIME, single_units_only: bool = True
) -> pd.DataFrame:
    """Best matching ground truth neuron of each sorted unit, and their agreement score."""
    if single_units_only:
        sorting = single_units(sorting)
    scores = agreement_scores(ground_truth, sorting, delta_time)
    return pd.DataFrame(
        {"ground_truth_unit": scores.idxmax(axis=0), "score": scores.max(axis=0)},
        index=scores.columns,
    ).rename_axis("unit_id")


def count_well_sorted(accuracy: pd.Series, threshold: float = WELL_SORTED) -> int:
    """Number of ground truth neurons sorted with an accuracy of at least threshold."""
    return int((accuracy >= threshold).sum())
