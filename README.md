# spikebias

[![tests](https://github.com/spikebias/spikebias/actions/workflows/tests.yml/badge.svg)](https://github.com/spikebias/spikebias/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/spikebias)](https://pypi.org/project/spikebias/)

Measure the biases of spike sorting on dense extracellular recordings, such as Neuropixels, against the
ground truth of a large-scale, detailed biophysical model of the rat cortex.

`spikebias` works on [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) extractors, so it
applies to any sorter and any recording with a ground truth. It also streams the simulated recordings,
ground truths and sortings of the paper from the DANDI Archive
([dandiset 001250](https://dandiarchive.org/dandiset/001250)).

## What it measures

| Module | Measure |
|---|---|
| `spikebias.accuracy` | Sorting accuracy of each ground truth neuron: its best agreement score with a sorted unit |
| `spikebias.biases` | Spike assignment bias of each sorted unit: good, overmerger, oversplitter, their mixtures, or false positive |
| `spikebias.sampling` | Yield (isolated units per recording site, by layer) and selection biases (neuron types over- or under-represented among the sorted units) |
| `spikebias.features`, `spikebias.model` | Model of the sorting accuracy of each neuron from its biological features (firing rate, layer, cell type, distance to the probe, spike amplitude and spatial extent) |
| `spikebias.datasets` | Access to the recordings, ground truths and sortings of dandiset 001250 |

## Install

```bash
pip install spikebias
```

Python 3.10 or later and SpikeInterface 0.101 or later.

## Quick start

```python
import spikebias as sb
from spikebias import datasets

# stream the ground truth and the Kilosort 4 sorting of the spontaneous Neuropixels simulation
ground_truth = sb.first_minutes(datasets.load_ground_truth("npx_spont"), minutes=10)
sorting = datasets.load_sorting("npx_spont", sorter="ks4")      # first 10 minutes

# sorting accuracy of each ground truth neuron
accuracy = sb.sorting_accuracy(ground_truth, sorting)
print(sb.count_well_sorted(accuracy), "of", len(accuracy), "neurons sorted with an accuracy >= 0.8")

# spike assignment bias of each sorted single-unit
biases = sb.classify_sorted_units(ground_truth, sorting, duration=600)
print(sb.bias_proportions(biases))

# yield per recording site, and selection biases by synapse class, layer and electrical type
recording = datasets.load_recording("npx_spont")                # streamed: only what is read is downloaded
print(sb.yield_per_site(sorting, recording))
print(sb.selection_biases(ground_truth, sorting))
```

Your own data work the same way: pass a SpikeInterface ground truth sorting and a sorting of the same
period. The sampling and accuracy-model functions use the neurons' properties (`layer`, `synapse_class`,
`etype`, `x`, `y`, `z`), which the biophysical datasets provide.

### Model of the sorting accuracy

```python
recording = sb.first_minutes(datasets.load_recording("npx_spont", local_folder="data"), minutes=10)
features = sb.neuron_features(ground_truth, recording, duration=600, n_jobs=8)
data = features.join(sb.sorting_accuracy(ground_truth, sorting)).dropna()   # drop neurons without spikes

model = sb.AccuracyModel(transform="rank")
r2 = sb.cross_validate(model, data.drop(columns="accuracy"), data["accuracy"], n_splits=100)
print(sb.summarize(r2))                       # McFadden pseudo R-squared on held-out neurons
print(model.fit(data.drop(columns="accuracy"), data["accuracy"]).weights)
```

`neuron_features` reads the spikes of every neuron from the recording, so download it first for large
datasets (`local_folder`); the Neuropixels recordings are about 100 GB.

## Definitions

- **Hit:** a sorted spike within `delta_time` = 1.3 ms of a ground truth spike.
- **Agreement score** of a ground truth neuron and a sorted unit: hits / (hits + misses + false positives).
- **Sorting accuracy** of a ground truth neuron: its best agreement score with a sorted (single-)unit; it is
  well sorted at 0.8 or more.
- **Chance score** of a pair: the agreement score expected from two independent Poisson spike trains with
  the units' firing rates.
- **Bias classes** of a sorted unit: *false positive* if no agreement score is above chance; otherwise *good*
  (a score of at least 0.8), *overmerger* (more than one neuron matched above chance), *oversplitter*
  (a matched neuron is also matched above chance by another sorted unit), or a *mixed* combination.

## Datasets

`datasets.DATASETS` lists the datasets of dandiset 001250, and `datasets.list_assets(name)` their files. All functions read the
published version 0.261108.1908 (`datasets.DANDISET_VERSION`) unless you pass `version="draft"`.

| Name | Content |
|---|---|
| `npx_spont`, `npx_evoked`, `npx_evoked_40khz` | Biophysical simulations, Neuropixels 1.0 (spontaneous and evoked activity) |
| `dense_probe1`, `dense_probe2`, `dense_probe3` | Biophysical simulations, 128-site dense probe at three depths |
| `npx_spont_disconnected` | Biophysical simulation of the disconnected circuit |
| `buccino`, `buccino_rep`, `mearec_s2` to `mearec_s5` | Synthetic MEArec models |
| `marques_smith`, `horvath_probe1` to `horvath_probe3` | In vivo recordings (no ground truth) with their sortings |

Sorters: `ks4`, `ks3`, `ks2.5`, `ks2`, `ks` (Kilosort 4 to 1) and `hs` (Herdingspikes), on the first
10 minutes (`minutes=10`) or, for some datasets, the full recording (`minutes=None`).

## Citation

If you use `spikebias` or the dataset, please cite:

- Laquitaine S, Imbeni M, Tharayil J, Isbister JB, Reimann MW. Spike sorting biases and information loss in
  a detailed cortical model. bioRxiv (2024). https://doi.org/10.1101/2024.12.04.626805
- The dataset: Laquitaine S, Imbeni M, Tharayil J, Isbister J, Reimann M. Spike sorting biases and information loss in a
  detailed cortical model (Version 0.261108.1908) [Data set]. DANDI Archive (2026). https://doi.org/10.48324/dandi.001250/0.261108.1908

The code that reproduces the paper's figures is at https://github.com/spikebias/spikebias-paper.
For the curation of sorted single-units, see [spikecurate](https://pypi.org/project/spikecurate/).

## License

GPL-3.0. Author: Steeve Laquitaine (laquitainesteeve@gmail.com).
