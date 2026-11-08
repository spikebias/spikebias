import numpy as np
import pytest
import spikeinterface as si

DURATION = 30.0
FS = 20000.0


@pytest.fixture(scope="session")
def simulation():
    """Synthetic recording and ground truth with the properties of the biophysical datasets."""
    recording, ground_truth = si.generate_ground_truth_recording(
        durations=[DURATION], sampling_frequency=FS, num_channels=16, num_units=8, seed=0
    )
    rng = np.random.default_rng(0)
    sites = np.c_[recording.get_channel_locations(), np.zeros(recording.get_num_channels())]
    for key, values in zip(("rel_x", "rel_y", "rel_z"), sites.T):
        recording.set_property(key, values)
    recording.set_property("layers", np.array(["5"] * 8 + ["6"] * 8))

    n = len(ground_truth.unit_ids)
    xyz = sites[rng.integers(0, len(sites), n)] + rng.normal(0, 10, (n, 3))
    for key, values in zip("xyz", xyz.T):
        ground_truth.set_property(key, values)
    ground_truth.set_property("layer", np.array(["5", "6"] * (n // 2)))
    ground_truth.set_property("synapse_class", np.array(["EXC", "EXC", "INH", "EXC"] * (n // 4)))
    ground_truth.set_property("etype", np.array(["cADpyr", "cADpyr", "bNAC", "cADpyr"] * (n // 4)))
    ground_truth.set_property("exc_mini_frequency", rng.uniform(0.01, 0.1, n))
    ground_truth.set_property("dynamics_holding_current", rng.uniform(-0.1, 0, n))
    return recording, ground_truth


def numpy_sorting(trains: dict, layers=None, labels=None):
    sorting = si.NumpySorting.from_unit_dict({k: np.sort(np.asarray(v, dtype="int64")) for k, v in trains.items()}, FS)
    if layers is not None:
        sorting.set_property("layer", np.asarray(layers))
    if labels is not None:
        sorting.set_property("KSLabel", np.asarray(labels))
    return sorting


def trains_of(sorting) -> dict:
    return {u: sorting.get_unit_spike_train(u) for u in sorting.unit_ids}
