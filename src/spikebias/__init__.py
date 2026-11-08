"""spikebias: spike sorting biases of dense extracellular recordings (e.g. Neuropixels), measured against
the ground truth of a large-scale detailed biophysical model of the cortex."""

__version__ = "0.1.1"

from . import datasets
from .accuracy import DELTA_TIME, WELL_SORTED, agreement_scores, best_matches, count_well_sorted, sorting_accuracy
from .biases import BIAS_CLASSES, bias_proportions, chance_scores, classify_sorted_units, label_units
from .features import distance_to_nearest_site, neuron_features, site_locations_3d, waveform_features
from .model import AccuracyModel, cross_validate, summarize
from .sampling import (
    count_cortical_sites,
    ground_truth_yield_per_site,
    neuron_types,
    selection_biases,
    yield_per_site,
)
from .utils import first_minutes, single_units, standardize_layers

__all__ = [
    "datasets",
    "DELTA_TIME",
    "WELL_SORTED",
    "agreement_scores",
    "best_matches",
    "count_well_sorted",
    "sorting_accuracy",
    "BIAS_CLASSES",
    "bias_proportions",
    "chance_scores",
    "classify_sorted_units",
    "label_units",
    "distance_to_nearest_site",
    "neuron_features",
    "site_locations_3d",
    "waveform_features",
    "AccuracyModel",
    "cross_validate",
    "summarize",
    "count_cortical_sites",
    "ground_truth_yield_per_site",
    "neuron_types",
    "selection_biases",
    "yield_per_site",
    "first_minutes",
    "single_units",
    "standardize_layers",
]
