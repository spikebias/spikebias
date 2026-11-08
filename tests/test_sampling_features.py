import numpy as np

import spikebias as sb
from conftest import DURATION, numpy_sorting, trains_of


def test_standardize_layers():
    assert list(sb.standardize_layers(["1", "2", "3", "L2_3", "4", "5", "6", "WM", "Outside", b"5"])) == [
        "L1", "L2/3", "L2/3", "L2/3", "L4", "L5", "L6", "Out", "Out", "L5"]


def test_first_minutes(simulation):
    recording, ground_truth = simulation
    assert sb.first_minutes(recording, minutes=0.25).get_num_samples() == 15 * 20000
    assert sb.first_minutes(recording, minutes=10).get_num_samples() == recording.get_num_samples()


def test_yields(simulation):
    recording, ground_truth = simulation
    sorting = numpy_sorting(trains_of(ground_truth), layers=["5"] * 6 + ["6"] * 2, labels=["good"] * 7 + ["mua"])
    df = sb.yield_per_site(sorting, recording).set_index("layer")
    assert df.loc["L5", "n_units"] == 6 and df.loc["L6", "n_units"] == 1
    assert np.isclose(df.loc["L5", "yield_per_site"], 6 / 8)
    gt = sb.ground_truth_yield_per_site(ground_truth, recording).set_index("layer")
    assert gt["n_neurons"].sum() == len(ground_truth.unit_ids)
    assert sb.count_cortical_sites(recording) == 16


def test_selection_biases_of_a_perfect_sorting(simulation):
    _, ground_truth = simulation
    sorting = numpy_sorting(trains_of(ground_truth))
    df = sb.selection_biases(ground_truth, sorting)
    assert np.allclose(df["ratio"], 1.0)
    assert (df["bias"] == "same").all()
    assert (df["count_ground_truth"] == df["count_sorted"]).all()


def test_neuron_features(simulation):
    recording, ground_truth = simulation
    df = sb.neuron_features(ground_truth, recording, duration=DURATION, n_jobs=1, progress_bar=False)
    assert len(df) == len(ground_truth.unit_ids)
    for column in ("firing_rate", "synapse_class", "etype_cADpyr", "layer_5", "distance", "snr", "spatial_extent",
                   "exc_mini_frequency", "dynamics_holding_current"):
        assert column in df
    assert (df["distance"] >= 0).all() and (df["snr"] > 0).all() and (df["spatial_extent"] >= 0).all()
    assert df[["layer_1", "layer_23", "layer_4", "layer_5", "layer_6"]].sum(axis=1).eq(1).all()
