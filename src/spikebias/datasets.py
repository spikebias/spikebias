"""Access to the recordings, ground truths and sortings of DANDI dandiset 001250.

Files are streamed from the archive by default (only the data read are downloaded); download() saves
a whole file locally. Ground truth neurons get their biophysical properties (layer, synapse class,
etype, 3D position, ...) from tables shipped with the package.

Example:
    >>> from spikebias import datasets
    >>> ground_truth = datasets.load_ground_truth("npx_spont")
    >>> sorting = datasets.load_sorting("npx_spont", sorter="ks4")
"""

import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd

DANDISET_ID = "001250"
# published version of the dandiset (https://doi.org/10.48324/dandi.001250/0.261108.1908); "draft" for the latest uploads
DANDISET_VERSION = "0.261108.1908"
API_URL = "https://api.dandiarchive.org/api"

# session names of the sorters in the dandiset
SORTERS = {"ks4": "ks4", "ks3": "ks3", "ks2.5": "ks25", "ks2": "ks2", "ks": "ks", "hs": "hs"}


@dataclass(frozen=True)
class Dataset:
    subject: str
    sampling_frequency: float
    probe: str
    model: str
    description: str
    ground_truth: bool = True
    cell_properties: bool = False


DATASETS = {
    "npx_spont": Dataset("sub-001-fitted", 40000.0, "neuropixels", "biophysical",
                         "spontaneous activity, Neuropixels 1.0, 384 sites", cell_properties=True),
    "npx_evoked": Dataset("sub-002-fitted", 20000.0, "neuropixels", "biophysical",
                          "stimulus-evoked activity, Neuropixels 1.0, 384 sites", cell_properties=True),
    "npx_evoked_40khz": Dataset("sub-npx-evoked-sf-40Hz-fitted", 40000.0, "neuropixels", "biophysical",
                                "stimulus-evoked activity sampled at 40 kHz, Neuropixels 1.0", cell_properties=True),
    "npx_spont_disconnected": Dataset("sub-biophy-npx-spont-disconnected-fitted", 40000.0, "neuropixels",
                                      "biophysical", "spontaneous activity of the disconnected circuit"),
    "dense_probe1": Dataset("sub-003-fitted", 20000.0, "dense", "biophysical",
                            "spontaneous activity, dense probe at depth 1 (layers 1 and 2/3), 128 sites", cell_properties=True),
    "dense_probe2": Dataset("sub-004-fitted", 20000.0, "dense", "biophysical",
                            "spontaneous activity, dense probe at depth 2 (layers 2 to 5), 128 sites", cell_properties=True),
    "dense_probe3": Dataset("sub-005-fitted", 20000.0, "dense", "biophysical",
                            "spontaneous activity, dense probe at depth 3 (layer 6), 128 sites", cell_properties=True),
    "buccino": Dataset("sub-mearec-synthetics-buccino", 32000.0, "neuropixels", "synthetic",
                       "sortings of the MEArec model of Buccino et al. (2020); recording and ground truth "
                       "in dandiset 000034", ground_truth=False),
    "buccino_rep": Dataset("sub-mearec-synthetics-buccino-rep", 32000.0, "neuropixels", "synthetic",
                           "reproduction of the model of Buccino et al. (2020), high-pass filtered at 100 Hz"),
    **{f"mearec_s{k}": Dataset(f"sub-mearec-synthetics-s{k}", 32000.0, "neuropixels", "synthetic",
                               f"MEArec synthetic model S{k}") for k in (2, 3, 4, 5)},
    "marques_smith": Dataset("sub-vivo-marques-smith", 30000.0, "neuropixels", "in vivo",
                             "in vivo recording of Marques-Smith et al. (2018)", ground_truth=False),
    **{f"horvath_probe{k}": Dataset(f"sub-vivo-horvath-depth-{k}", 20000.0, "dense", "in vivo",
                                    f"in vivo recording of Horvath et al. (2021), depth {k}", ground_truth=False)
       for k in (1, 2, 3)},
}


def _dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]


def _get(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def asset_path(name: str, sorter: str = None, minutes: int = 10, session: str = None) -> str:
    """Path of a file in the dandiset: the recording and ground truth file, or a sorting."""
    subject = _dataset(name).subject
    if session is None and sorter is None:
        return f"{subject}/{subject}_ecephys.nwb"
    if session is None:
        if sorter not in SORTERS:
            raise KeyError(f"unknown sorter {sorter!r}; choose from {list(SORTERS)}")
        session = SORTERS[sorter] + (f"-{minutes}m" if minutes else "")
    return f"{subject}/{subject}_ses-{session}.nwb"


def list_assets(name: str = None, version: str = DANDISET_VERSION) -> pd.DataFrame:
    """Files of the dandiset (of one dataset if name is given), with their size in GB."""
    rows, url = [], f"{API_URL}/dandisets/{DANDISET_ID}/versions/{version}/assets/?page_size=500"
    while url:
        page = _get(url)
        rows += [{"path": a["path"], "size_gb": a["size"] / 1e9} for a in page["results"]]
        url = page.get("next")
    df = pd.DataFrame(rows).sort_values("path", ignore_index=True)
    if name is not None:
        df = df[df["path"].str.startswith(_dataset(name).subject + "/")].reset_index(drop=True)
    return df


def asset_url(path: str, version: str = DANDISET_VERSION) -> str:
    """Direct (S3) URL of a file of the dandiset."""
    page = _get(f"{API_URL}/dandisets/{DANDISET_ID}/versions/{version}/assets/?path={path}")
    match = [a for a in page["results"] if a["path"] == path]
    if not match:
        raise FileNotFoundError(f"{path} is not in dandiset {DANDISET_ID} ({version})")
    metadata = _get(f"{API_URL}/dandisets/{DANDISET_ID}/versions/{version}/assets/{match[0]['asset_id']}/")
    urls = [u for u in metadata["contentUrl"] if "amazonaws" in u]
    return urls[0] if urls else metadata["contentUrl"][0]


def download(path: str, folder: str = "spikebias_data", version: str = DANDISET_VERSION) -> str:
    """Download a file of the dandiset (e.g. asset_path("npx_spont")) and return its local path."""
    local = os.path.join(folder, path)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with urllib.request.urlopen(asset_url(path, version)) as response, open(local + ".part", "wb") as f:
            shutil.copyfileobj(response, f, length=16 * 1024 * 1024)
        os.replace(local + ".part", local)
    return local


def _source(path: str, local_folder: str, version: str) -> dict:
    if local_folder is not None:
        return {"file_path": download(path, local_folder, version)}
    return {"file_path": asset_url(path, version), "stream_mode": "remfile"}


def cell_properties(name: str) -> pd.DataFrame:
    """Biophysical properties of the ground truth neurons of a biophysical dataset, indexed by unit id."""
    if not _dataset(name).cell_properties:
        raise KeyError(f"no cell properties for {name!r}")
    with resources.as_file(resources.files("spikebias") / "data" / f"cell_properties_{name}.csv.gz") as f:
        return pd.read_csv(f, index_col="unit_id")


def add_cell_properties(ground_truth, name: str):
    """Set the biophysical properties of the ground truth neurons as properties of the extractor."""
    df = cell_properties(name)
    df = df.loc[np.asarray(ground_truth.unit_ids).astype(int)]
    for key in df.columns:
        values = df[key].values
        ground_truth.set_property(key, values.astype(str) if values.dtype == object else values)
    return ground_truth


def _site_coordinates(source: dict) -> np.ndarray:
    """3D coordinates of the recording sites (rel_x, rel_y, rel_z of the NWB electrodes table)."""
    import h5py

    file = source["file_path"]
    if source.get("stream_mode") == "remfile":
        import remfile

        file = remfile.File(file)
    with h5py.File(file, "r") as f:
        table = f["general/extracellular_ephys/electrodes"]
        xyz = np.c_[table["rel_x"][:], table["rel_y"][:], table["rel_z"][:]]
        series = [f["acquisition"][k] for k in f["acquisition"] if "electrodes" in f["acquisition"][k]]
        rows = series[0]["electrodes"][:] if series else np.arange(len(xyz))
    return xyz[rows]


def load_recording(name: str, local_folder: str = None, version: str = DANDISET_VERSION):
    """RecordingExtractor of a dataset (traces in uV), streamed unless local_folder is given.

    The 3D site coordinates, in the frame of the neurons' x, y, z, are set as the rel_x, rel_y and
    rel_z properties (the probe itself holds 2D coordinates).
    """
    from spikeinterface.extractors import read_nwb_recording

    source = _source(asset_path(name), local_folder, version)
    recording = read_nwb_recording(**source)
    xyz = _site_coordinates(source)
    if len(xyz) == recording.get_num_channels():
        for key, values in zip(("rel_x", "rel_y", "rel_z"), xyz.T):
            recording.set_property(key, values)
    return recording


def load_ground_truth(name: str, local_folder: str = None, version: str = DANDISET_VERSION):
    """SortingExtractor of the ground truth neurons of a dataset, with their properties."""
    from spikeinterface.extractors import read_nwb_sorting

    dataset = _dataset(name)
    if not dataset.ground_truth:
        raise KeyError(f"{name!r} has no ground truth in dandiset {DANDISET_ID}")
    sorting = read_nwb_sorting(**_source(asset_path(name), local_folder, version),
                               sampling_frequency=dataset.sampling_frequency, t_start=0)
    if dataset.cell_properties:
        sorting = add_cell_properties(sorting, name)
    return sorting


def load_sorting(name: str, sorter: str = "ks4", minutes: int = 10, session: str = None,
                 local_folder: str = None, version: str = DANDISET_VERSION):
    """SortingExtractor of a spike sorter on a dataset (the first 10 minutes by default).

    Args:
        name: dataset name (see DATASETS)
        sorter: "ks4", "ks3", "ks2.5", "ks2", "ks" or "hs"
        minutes: 10 for the first 10 minutes, None for the full recording (when available)
        session: session name instead of sorter and minutes, e.g. "ks4-v100-10m" or "tc"
        local_folder: download the file to this folder instead of streaming it
        version: dandiset version (default: the published version DANDISET_VERSION)
    """
    from spikeinterface.extractors import read_nwb_sorting

    path = asset_path(name, sorter=sorter, minutes=minutes, session=session)
    return read_nwb_sorting(**_source(path, local_folder, version),
                            sampling_frequency=_dataset(name).sampling_frequency, t_start=0)
