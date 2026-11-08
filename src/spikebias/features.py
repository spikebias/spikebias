"""Biological and recording features of ground truth neurons, used to model their sorting accuracy."""

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.preprocessing as spre

from .utils import get_duration, get_traces_uv, spike_counts, standardize_layers

LAYER_COLUMNS = {"L1": "layer_1", "L2/3": "layer_23", "L4": "layer_4", "L5": "layer_5", "L6": "layer_6"}


def site_locations_3d(recording) -> np.ndarray:
    """3D coordinates of the recording sites (um), in the frame of the neurons' x, y, z properties."""
    keys = recording.get_property_keys()
    if all(k in keys for k in ("rel_x", "rel_y", "rel_z")):
        return np.column_stack([np.asarray(recording.get_property(k), dtype=float) for k in ("rel_x", "rel_y", "rel_z")])
    location = recording.get_property("location")
    if location is not None and np.asarray(location).shape[1] == 3:
        return np.asarray(location, dtype=float)
    raise ValueError("the recording has no 3D site coordinates: pass site_locations")


def distance_to_nearest_site(ground_truth, site_locations) -> np.ndarray:
    """3D distance of each ground truth neuron (x, y, z properties) to its nearest recording site (um)."""
    neurons = np.column_stack([np.asarray(ground_truth.get_property(k), dtype=float) for k in ("x", "y", "z")])
    sites = np.asarray(site_locations, dtype=float)
    return np.sqrt(((neurons[:, None, :] - sites[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def preprocess(recording, freq_min: float = 300):
    """High-pass filter and subtract the median of all sites, as before sorting in the paper."""
    recording = spre.highpass_filter(recording, freq_min=freq_min)
    return spre.common_reference(recording, reference="global", operator="median")


def noise_levels(recording, num_chunks: int = 20, chunk_size: int = 10000, seed: int = 0) -> np.ndarray:
    """Noise level of each site (uV): median absolute deviation of random chunks of traces, / 0.6745."""
    n_samples = recording.get_num_samples(segment_index=0)
    chunk_size = min(chunk_size, n_samples)
    starts = np.random.default_rng(seed).integers(0, n_samples - chunk_size + 1, size=num_chunks)
    traces = np.concatenate([get_traces_uv(recording, int(s), int(s) + chunk_size) for s in starts])
    return np.median(np.abs(traces - np.median(traces, axis=0)), axis=0) / 0.6744897501960817


def waveform_features(
    ground_truth,
    recording,
    site_locations=None,
    ms_before: float = 3,
    ms_after: float = 3,
    max_spikes_per_unit: int = 500,
    radius_um: float = 100,
    seed: int = 0,
    filter_recording: bool = True,
    **job_kwargs,
) -> pd.DataFrame:
    """Signal-to-noise ratio and spatial extent of each ground truth neuron's average spike.

    The largest site is the one where the average spike has its most negative peak. The snr is the
    amplitude of that peak divided by the site's noise level (median absolute deviation). The spatial
    extent sums the 3D distances to the largest site of the sites within radius_um of it, weighted by the
    cosine similarity (clipped at 0) of their average spike with the largest site's.

    Args:
        ground_truth: SortingExtractor of the ground truth neurons
        recording: RecordingExtractor (raw; filtered here unless filter_recording is False)
        site_locations: 3D site coordinates (default: site_locations_3d(recording))
        ms_before, ms_after: spike window (ms)
        max_spikes_per_unit: number of spikes averaged per neuron
        radius_um: radius of the neuron's sites around its largest site, on the probe (um)
        seed: seed of the spike selection and of the noise estimate
        job_kwargs: SpikeInterface job arguments (n_jobs, chunk_duration, progress_bar)

    Returns:
        pd.DataFrame: snr and spatial_extent, indexed by unit id (NaN for neurons without spikes)
    """
    sites = site_locations_3d(recording) if site_locations is None else np.asarray(site_locations, dtype=float)
    if filter_recording:
        recording = preprocess(recording)
    spiking = ground_truth.unit_ids[spike_counts(ground_truth) > 0]
    previous = si.get_global_job_kwargs()
    si.set_global_job_kwargs(**job_kwargs)
    try:
        analyzer = si.create_sorting_analyzer(ground_truth.select_units(spiking), recording, format="memory", sparse=False)
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=max_spikes_per_unit, seed=seed)
        analyzer.compute("templates", ms_before=ms_before, ms_after=ms_after)
    finally:
        si.set_global_job_kwargs(**previous)
    templates = analyzer.get_extension("templates").get_templates(operator="average")
    noise = noise_levels(recording, seed=seed)
    probe_xy = recording.get_channel_locations()

    snr, extent = {}, {}
    for ix, unit_id in enumerate(analyzer.unit_ids):
        largest = int(np.argmin(templates[ix].min(axis=0)))
        snr[unit_id] = float(abs(templates[ix][:, largest].min()) / noise[largest])
        channels = np.flatnonzero(np.linalg.norm(probe_xy - probe_xy[largest], axis=1) <= radius_um)
        spikes, reference = templates[ix][:, channels], templates[ix][:, largest]
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = reference @ spikes / (np.linalg.norm(spikes, axis=0) * np.linalg.norm(reference))
        distance = np.linalg.norm(sites[channels] - sites[largest], axis=1)
        extent[unit_id] = float(np.nansum(np.clip(similarity, 0, None) * distance))

    df = pd.DataFrame(index=pd.Index(ground_truth.unit_ids, name="unit_id"))
    df["snr"] = pd.Series(snr)
    df["spatial_extent"] = pd.Series(extent)
    return df


def neuron_features(
    ground_truth,
    recording=None,
    duration: float = None,
    site_locations=None,
    waveforms: bool = True,
    **waveform_kwargs,
) -> pd.DataFrame:
    """Features of each ground truth neuron for the sorting accuracy model.

    firing_rate, synapse_class (1 if excitatory), one column per electrical type and per layer,
    distance to the nearest site (3D), snr and spatial_extent of the average spike (if waveforms),
    exc_mini_frequency and dynamics_holding_current.

    Args:
        ground_truth: SortingExtractor of the ground truth neurons, with their properties
        recording: RecordingExtractor of the same period (for the distance and the waveforms)
        duration: duration of the period (s); inferred from the ground truth when possible
        site_locations: 3D site coordinates (default: site_locations_3d(recording))
        waveforms: compute snr and spatial_extent (slow: it reads the recording)
        waveform_kwargs: arguments of waveform_features

    Returns:
        pd.DataFrame: one row per ground truth neuron
    """
    df = pd.DataFrame(index=pd.Index(ground_truth.unit_ids, name="unit_id"))
    df["firing_rate"] = spike_counts(ground_truth) / get_duration(ground_truth, duration)
    df["synapse_class"] = (np.asarray(ground_truth.get_property("synapse_class")).astype(str) == "EXC").astype(int)

    etypes = pd.get_dummies(np.asarray(ground_truth.get_property("etype")).astype(str), prefix="etype", dtype=int)
    df[etypes.columns] = etypes.values
    layers = pd.Categorical(standardize_layers(ground_truth.get_property("layer")), categories=list(LAYER_COLUMNS))
    df[list(LAYER_COLUMNS.values())] = pd.get_dummies(layers, dtype=int).values

    if recording is not None or site_locations is not None:
        sites = site_locations_3d(recording) if site_locations is None else site_locations
        df["distance"] = distance_to_nearest_site(ground_truth, sites)
    if waveforms:
        if recording is None:
            raise ValueError("waveforms=True needs the recording")
        df = df.join(waveform_features(ground_truth, recording, site_locations, **waveform_kwargs))
    for key in ("exc_mini_frequency", "dynamics_holding_current"):
        if key in ground_truth.get_property_keys():
            df[key] = np.asarray(ground_truth.get_property(key), dtype=float)
    return df
