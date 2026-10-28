"""Tests for weighted dataset sampling in BatchedMultisetSampler / MixedWellDataModule."""
import collections

import pytest
import torch
from torch.utils.data import RandomSampler

from walrus.data.mixed_dset_sampler import BatchedMultisetSampler
from walrus.data.multidatamodule import MixedWellDataModule


# ---------------------------------------------------------------------------
# Minimal mock dataset — no I/O, just enough for the sampler to work
# ---------------------------------------------------------------------------

class _MockSubDataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _MockMixedDataset:
    def __init__(self, sizes):
        self.sub_dsets = [_MockSubDataset(s) for s in sizes]
        # Mirror multidataset.py convention: offsets[0] = -1, rest are cumulative
        self.offsets = [-1] + [sum(sizes[: i + 1]) for i in range(len(sizes))]
        self.effective_batch_sizes = [1] * len(sizes)


def _run_sampler(sizes, weights, max_samples=2000):
    """Run sampler and return per-dataset batch counts."""
    mock = _MockMixedDataset(sizes)
    wt = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    sampler = BatchedMultisetSampler(
        mock,
        RandomSampler,
        batch_size=1,
        max_samples=max_samples,
        recycle=True,
        dataset_weights=wt,
    )
    # offsets[1] is the start of dataset 1 in the global index space
    boundaries = [0] + [sum(sizes[: i + 1]) for i in range(len(sizes))]
    counts = collections.Counter()
    for batch in sampler:
        idx = batch[0]  # each yielded item is a list of one global index
        for d in range(len(sizes) - 1, -1, -1):
            if idx >= boundaries[d]:
                counts[d] += 1
                break
    return counts


# ---------------------------------------------------------------------------
# Sampler unit tests (no I/O)
# ---------------------------------------------------------------------------

def test_no_weights_uniform_two_datasets():
    counts = _run_sampler([500, 500], weights=None)
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.5) < 0.05, counts


def test_weights_75_25():
    counts = _run_sampler([500, 500], weights=[0.75, 0.25])
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.75) < 0.05, counts


def test_weights_need_not_sum_to_one():
    """[3, 1] is the same ratio as [0.75, 0.25]."""
    counts = _run_sampler([500, 500], weights=[3.0, 1.0])
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.75) < 0.05, counts


def test_weights_equal_explicit_same_as_uniform():
    counts = _run_sampler([500, 500], weights=[1.0, 1.0])
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.5) < 0.05, counts


def test_weights_three_datasets():
    counts = _run_sampler([200, 200, 200], weights=[0.5, 0.3, 0.2], max_samples=3000)
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.5) < 0.05, counts
    assert abs(counts[1] / total - 0.3) < 0.05, counts
    assert abs(counts[2] / total - 0.2) < 0.05, counts


def test_count_respects_max_samples():
    mock = _MockMixedDataset([500, 500])
    sampler = BatchedMultisetSampler(
        mock, RandomSampler, batch_size=1, max_samples=100, recycle=True,
        dataset_weights=torch.tensor([0.75, 0.25]),
    )
    assert sum(1 for _ in sampler) == 100


# ---------------------------------------------------------------------------
# MixedWellDataModule integration tests (uses dummy_dataset fixture)
# ---------------------------------------------------------------------------

def test_datamodule_no_weights_gives_none(dummy_dataset):
    dm = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {"path": dummy_dataset / "dummy"},
        },
        batch_size=1,
        data_workers=0,
        max_samples=5,
        prefetch_field_names=False,
    )
    assert dm.dataset_weights is None


def test_datamodule_extracts_single_weight(dummy_dataset):
    """Even one dataset with sampling_weight should populate the tensor."""
    dm = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {"path": dummy_dataset / "dummy", "sampling_weight": 0.8},
        },
        batch_size=1,
        data_workers=0,
        max_samples=5,
        prefetch_field_names=False,
    )
    assert dm.dataset_weights is not None
    assert abs(dm.dataset_weights[0].item() - 0.8) < 1e-6


def test_datamodule_extracts_two_weights(dummy_dataset):
    dm = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy_a": {"path": dummy_dataset / "dummy", "sampling_weight": 0.75},
            "dummy_b": {"path": dummy_dataset / "dummy", "sampling_weight": 0.25},
        },
        batch_size=1,
        data_workers=0,
        max_samples=5,
        prefetch_field_names=False,
    )
    assert dm.dataset_weights is not None
    assert abs(dm.dataset_weights[0].item() - 0.75) < 1e-6
    assert abs(dm.dataset_weights[1].item() - 0.25) < 1e-6


def test_datamodule_partial_weights_default_to_one(dummy_dataset):
    """A dataset without sampling_weight should default to weight 1.0."""
    dm = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy_a": {"path": dummy_dataset / "dummy", "sampling_weight": 2.0},
            "dummy_b": {"path": dummy_dataset / "dummy"},  # no weight
        },
        batch_size=1,
        data_workers=0,
        max_samples=5,
        prefetch_field_names=False,
    )
    assert dm.dataset_weights is not None
    assert abs(dm.dataset_weights[0].item() - 2.0) < 1e-6
    assert abs(dm.dataset_weights[1].item() - 1.0) < 1e-6


def test_datamodule_train_dataloader_runs_with_weights(dummy_dataset):
    """End-to-end: weighted datamodule should produce correct number of batches."""
    dm = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {"path": dummy_dataset / "dummy", "sampling_weight": 1.0},
        },
        batch_size=1,
        data_workers=0,
        max_samples=10,
        prefetch_field_names=False,
    )
    batches = list(dm.train_dataloader())
    assert len(batches) == 10
