"""Sampling biases: how many neurons are isolated per recording site, and which neuron types."""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .accuracy import DELTA_TIME, best_matches
from .utils import single_units, spike_counts, standardize_layers

NEURON_TYPE = ["synapse_class", "layer", "etype"]


def site_layers(recording, layers=None) -> np.ndarray:
    """Layer of each recording site, from the "layers" (or "layer") property unless given."""
    if layers is not None:
        return standardize_layers(layers)
    for key in ("layers", "layer"):
        if key in recording.get_property_keys():
            return standardize_layers(recording.get_property(key))
    raise ValueError("the recording has no site layers: pass layers")


def count_cortical_sites(recording, layers=None) -> int:
    """Number of recording sites in the cortex."""
    return int((site_layers(recording, layers) != "Out").sum())


def _per_site(unit_layers, recording, layers, name) -> pd.DataFrame:
    units = pd.Series(unit_layers, name="layer").value_counts().rename(name)
    sites = pd.Series(site_layers(recording, layers), name="layer").value_counts().rename("n_sites")
    df = pd.concat([units, sites], axis=1, join="inner").rename_axis("layer").reset_index()
    df["yield_per_site"] = df[name] / df["n_sites"]
    return df.sort_values("layer", ignore_index=True)


def yield_per_site(sorting, recording, single_units_only: bool = True, layers=None) -> pd.DataFrame:
    """Number of sorted (single-)units per recording site in each layer.

    The layer of a sorted unit is its "layer" property (the layer of its nearest site).

    Args:
        sorting: SortingExtractor of the spike sorter
        recording: RecordingExtractor, with the layer of each site
        single_units_only: count only the sorter's single-units (Kilosort "good" units)
        layers: layer of each site, if the recording has no "layers" property

    Returns:
        pd.DataFrame: layer, n_units, n_sites and yield_per_site
    """
    if single_units_only:
        sorting = single_units(sorting)
    return _per_site(standardize_layers(sorting.get_property("layer")), recording, layers, "n_units")


def ground_truth_yield_per_site(ground_truth, recording, layers=None) -> pd.DataFrame:
    """Number of spiking ground truth neurons per recording site in each layer (the maximal yield)."""
    spiking = spike_counts(ground_truth) > 0
    unit_layers = standardize_layers(ground_truth.get_property("layer"))[spiking]
    return _per_site(unit_layers, recording, layers, "n_neurons")


def neuron_types(ground_truth, features=NEURON_TYPE) -> pd.DataFrame:
    """Type of each ground truth neuron (by default its synapse class, layer and electrical type)."""
    df = pd.DataFrame({key: np.asarray(ground_truth.get_property(key)) for key in features}, index=ground_truth.unit_ids)
    if "layer" in df:
        df["layer"] = standardize_layers(df["layer"])
    return df.astype(str).rename_axis("unit_id")


def selection_biases(
    ground_truth,
    sorting,
    delta_time: float = DELTA_TIME,
    single_units_only: bool = True,
    features=NEURON_TYPE,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Over- and under-representation of each neuron type among the sorted units.

    Each sorted unit takes the type of its best matching ground truth neuron, except for its layer,
    which is the sorted unit's own "layer" property when it exists. For each type, the ratio is its
    proportion among the sorted units divided by its proportion in the ground truth, and the p-value
    is that of a two-sided Fisher exact test.

    Returns:
        pd.DataFrame: one row per neuron type, with count_ground_truth, count_sorted, ratio, p_value
        and bias ("increased", "reduced" or "same" at the alpha level)
    """
    if single_units_only:
        sorting = single_units(sorting)
    types_gt = neuron_types(ground_truth, features)
    matches = best_matches(ground_truth, sorting, delta_time, single_units_only=False)
    types_sorted = types_gt.loc[matches["ground_truth_unit"].values].reset_index(drop=True)
    if "layer" in features and "layer" in sorting.get_property_keys():
        types_sorted["layer"] = standardize_layers(sorting.get_property("layer"))

    features = list(features)
    sort_by = [f for f in ("synapse_class", "layer") if f in features] or features
    unique = types_gt.drop_duplicates().sort_values(by=sort_by).reset_index(drop=True)
    count_gt = types_gt.value_counts().reindex(pd.MultiIndex.from_frame(unique), fill_value=0).values
    count_sorted = types_sorted.value_counts().reindex(pd.MultiIndex.from_frame(unique), fill_value=0).values

    ratio = (count_sorted / count_sorted.sum()) / (count_gt / count_gt.sum())
    p_values = np.array([
        fisher_exact([[g, s], [count_gt.sum() - g, count_sorted.sum() - s]], alternative="two-sided")[1]
        for g, s in zip(count_gt, count_sorted)
    ])
    df = unique.assign(count_ground_truth=count_gt, count_sorted=count_sorted, ratio=ratio, p_value=p_values)
    df["bias"] = np.where(df["p_value"] < alpha, np.where(df["ratio"] > 1, "increased", "reduced"), "same")
    return df
