import os

import numpy as np
import pandas as pd
import pytest

import spikebias as sb
from spikebias import datasets
from conftest import numpy_sorting


def synthetic_dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "firing_rate": rng.lognormal(0, 1, n),
        "snr": rng.lognormal(1, 0.5, n),
        "distance": rng.uniform(0, 50, n),
        "synapse_class": rng.integers(0, 2, n),
    })
    logit = 1.5 * np.log(X["snr"]) - 0.05 * X["distance"] + 0.5 * X["synapse_class"]
    y = np.clip(1 / (1 + np.exp(-logit)) + rng.normal(0, 0.05, n), 0, 1)
    return X, y


@pytest.mark.parametrize("transform", ["rank", "zscore"])
def test_accuracy_model(transform):
    X, y = synthetic_dataset()
    model = sb.AccuracyModel(transform=transform)
    r2 = sb.cross_validate(model, X, y, n_splits=10)
    summary = sb.summarize(r2)
    assert summary["n_splits"] == 10 and summary["median"] > 0.05
    model.fit(X, y)
    assert set(model.weights.index) == {"const", *X.columns}
    assert model.weights.loc["snr", "weight"] > 0 and model.weights.loc["distance", "weight"] < 0
    prediction = model.predict(X.iloc[:5] * 100)
    assert ((prediction >= 0) & (prediction <= 1)).all()


def test_model_rejects_missing_features():
    X, y = synthetic_dataset()
    X.loc[0, "snr"] = np.nan
    with pytest.raises(ValueError):
        sb.AccuracyModel().fit(X, y)


def test_asset_paths():
    assert datasets.asset_path("npx_spont") == "sub-001-fitted/sub-001-fitted_ecephys.nwb"
    assert datasets.asset_path("npx_evoked", sorter="ks2.5") == "sub-002-fitted/sub-002-fitted_ses-ks25-10m.nwb"
    assert datasets.asset_path("marques_smith", sorter="ks4", minutes=None) == "sub-vivo-marques-smith/sub-vivo-marques-smith_ses-ks4.nwb"
    assert datasets.asset_path("npx_spont", session="ks4-v100-10m") == "sub-001-fitted/sub-001-fitted_ses-ks4-v100-10m.nwb"
    with pytest.raises(KeyError):
        datasets.asset_path("unknown")


@pytest.mark.parametrize("name,n", [("npx_spont", 1388), ("npx_evoked", 1836), ("npx_evoked_40khz", 1552),
                                    ("dense_probe1", 287), ("dense_probe2", 770), ("dense_probe3", 1123)])
def test_cell_properties(name, n):
    df = datasets.cell_properties(name)
    assert len(df) == n
    assert {"layer", "synapse_class", "etype", "x", "y", "z"} <= set(df.columns)


def test_add_cell_properties():
    df = datasets.cell_properties("dense_probe1")
    ids = [str(u) for u in df.index[:3]]
    sorting = numpy_sorting({u: [10, 20] for u in ids})
    datasets.add_cell_properties(sorting, "dense_probe1")
    assert list(sorting.get_property("etype")) == list(df["etype"].iloc[:3])


network = pytest.mark.skipif(os.environ.get("SPIKEBIAS_NETWORK") != "1", reason="set SPIKEBIAS_NETWORK=1")


@pytest.mark.network
@network
def test_stream_sorting_and_ground_truth():
    sorting = datasets.load_sorting("npx_spont", sorter="ks4")
    assert len(sorting.unit_ids) == 520 and "KSLabel" in sorting.get_property_keys()
    ground_truth = datasets.load_ground_truth("dense_probe1")
    assert len(ground_truth.unit_ids) == 287 and "synapse_class" in ground_truth.get_property_keys()
    assert "sub-003-fitted/sub-003-fitted_ses-ks4-10m.nwb" in set(datasets.list_assets("dense_probe1")["path"])
