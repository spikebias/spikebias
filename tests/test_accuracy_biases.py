import numpy as np

import spikebias as sb
from conftest import DURATION, numpy_sorting, trains_of


def test_perfect_sorting_has_accuracy_one(simulation):
    _, ground_truth = simulation
    sorting = numpy_sorting(trains_of(ground_truth))
    accuracy = sb.sorting_accuracy(ground_truth, sorting)
    assert np.allclose(accuracy.values, 1.0)
    assert sb.count_well_sorted(accuracy) == len(ground_truth.unit_ids)


def test_single_units_only(simulation):
    _, ground_truth = simulation
    trains = trains_of(ground_truth)
    labels = ["good" if k % 2 == 0 else "mua" for k in range(len(trains))]
    sorting = numpy_sorting(trains, labels=labels)
    accuracy = sb.sorting_accuracy(ground_truth, sorting)
    assert np.allclose(accuracy.values[::2], 1.0)
    assert (accuracy.values[1::2] < 0.8).all()
    assert np.allclose(sb.sorting_accuracy(ground_truth, sorting, single_units_only=False).values, 1.0)


def test_best_matches(simulation):
    _, ground_truth = simulation
    sorting = numpy_sorting({f"s{u}": t for u, t in trains_of(ground_truth).items()})
    matches = sb.best_matches(ground_truth, sorting)
    assert list(matches["ground_truth_unit"]) == list(ground_truth.unit_ids)


def test_chance_scores_formula():
    chance = sb.chance_scores([10.0], [5.0], duration=600, delta_time=1.3)
    p_hit = 1 - np.exp(-2 * 1.3 * 5 / 1000)
    n_hits = p_hit * 5 * 600
    assert np.isclose(chance[0, 0], n_hits / (10 * 600 + 5 * 600 - n_hits))
    assert sb.chance_scores([0.0], [0.0], 600).item() == 0.0


def test_label_units():
    scores = np.array([[0.9, 0.0, 0.0], [0.5, 0.4, 0.0], [0.0, 0.0, 0.01], [0.0, 0.0, 0.0], [0.0, 0.0, 0.85]])
    labels = sb.label_units(scores, np.full(scores.shape, 0.05))
    assert list(labels) == [
        "mixed: good + oversplitter",
        "mixed: overmerger + oversplitter",
        "false positive",
        "false positive",
        "good",
    ]


def test_classify_merged_and_split_units(simulation):
    _, ground_truth = simulation
    trains = trains_of(ground_truth)
    units = list(trains)
    sorted_trains = {
        "merged": np.concatenate([trains[units[0]], trains[units[1]]]),
        "half_a": trains[units[2]][::2],
        "half_b": trains[units[2]][1::2],
        "copy": trains[units[3]],
        "noise": np.random.default_rng(1).integers(0, int(DURATION * 20000), 50),
    }
    sorting = numpy_sorting(sorted_trains, layers=["5", "5", "6", "6", "6"])
    biases = sb.classify_sorted_units(ground_truth, sorting, duration=DURATION)
    assert "overmerger" in biases.loc["merged", "bias"]
    assert "oversplitter" in biases.loc["half_a", "bias"] and "oversplitter" in biases.loc["half_b", "bias"]
    assert "good" in biases.loc["copy", "bias"]
    assert list(biases["layer"]) == ["L5", "L5", "L6", "L6", "L6"]

    proportions = sb.bias_proportions(biases)
    assert np.isclose(proportions["proportion"].sum(), 1.0)
    assert np.allclose(sb.bias_proportions(biases, by="layer").sum(axis=1), 1.0)
