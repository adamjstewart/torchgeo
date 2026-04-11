# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import pytest
from pandas import Timedelta, Timestamp
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.samplers import RandomTimestampSampler, SequentialTimestampSampler

TMIN = Timestamp(2025, 4, 1)
TMAX = Timestamp(2025, 4, 30)


class TestRandomTimestampSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> RandomTimestampSampler:
        return RandomTimestampSampler(dataset)

    def test_iter(self, sampler: RandomTimestampSampler) -> None:
        _, _, t = next(iter(sampler))
        assert TMIN <= t.start < t.stop <= TMAX
        assert t.stop - t.start == Timedelta('1D')

    def test_len(self, sampler: RandomTimestampSampler) -> None:
        assert len(sampler) == 3

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: RandomTimestampSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestSequentialTimestampSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> SequentialTimestampSampler:
        return SequentialTimestampSampler(dataset)

    def test_iter(self, sampler: SequentialTimestampSampler) -> None:
        _, _, t = next(iter(sampler))
        assert TMIN <= t.start < t.stop <= TMAX
        assert t.stop - t.start == Timedelta('1D')

    def test_len(self, sampler: SequentialTimestampSampler) -> None:
        assert len(sampler) == 3

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: SequentialTimestampSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
