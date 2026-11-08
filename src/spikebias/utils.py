"""Helpers shared by the modules."""

import numpy as np

# names of the cortical layers used in the analyses
LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]

_LAYER_NAMES = {
    "1": "L1",
    "2": "L2/3",
    "3": "L2/3",
    "L2": "L2/3",
    "L3": "L2/3",
    "L2_3": "L2/3",
    "4": "L4",
    "5": "L5",
    "6": "L6",
    "WM": "Out",
    "Outside": "Out",
}


def standardize_layers(layers) -> np.ndarray:
    """Map layer labels ("1", "2", "3", "L2_3", "WM", ...) to L1, L2/3, L4, L5, L6 or Out."""
    layers = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(layers).ravel()]
    return np.array([_LAYER_NAMES.get(x, x) for x in layers], dtype=object)


def single_units(sorting, label_property: str = "KSLabel", label: str = "good"):
    """Select the single-units of a sorting (Kilosort's "good" units).

    Sortings without the label property (e.g. Kilosort 1, Herdingspikes) are returned unchanged.
    """
    if label_property not in sorting.get_property_keys():
        return sorting
    labels = np.asarray(sorting.get_property(label_property)).astype(str)
    return sorting.select_units(sorting.unit_ids[labels == label])


def first_minutes(extractor, minutes: float = 10):
    """Keep the first minutes of a recording or sorting (the paper analyses the first 10 minutes)."""
    end_frame = int(minutes * 60 * extractor.get_sampling_frequency())
    try:
        end_frame = min(end_frame, int(extractor.get_num_samples(segment_index=0)))
    except Exception:
        pass
    return extractor.frame_slice(start_frame=0, end_frame=end_frame)


def get_duration(extractor, duration=None) -> float:
    """Duration in seconds: the one given, else the extractor's (sortings need a known duration)."""
    if duration is not None:
        return float(duration)
    try:
        return float(extractor.get_total_duration())
    except Exception as error:
        raise ValueError("cannot infer the duration of the sorting: pass duration (in seconds)") from error


def spike_counts(sorting) -> np.ndarray:
    """Number of spikes of each unit, in the order of sorting.unit_ids."""
    counts = sorting.count_num_spikes_per_unit()
    return np.array([counts[unit_id] for unit_id in sorting.unit_ids], dtype=float)


def get_traces_uv(recording, start_frame: int, end_frame: int) -> np.ndarray:
    """Traces in uV (the scaling argument was renamed across SpikeInterface versions)."""
    try:
        return recording.get_traces(start_frame=start_frame, end_frame=end_frame, return_in_uV=True)
    except TypeError:
        return recording.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=True)
