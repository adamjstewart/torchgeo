# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import pytest
from pandas import Interval, Timestamp
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.samplers import RandomTimestampSampler

TMIN = Timestamp(2025, 4, 1)
TMAX = Timestamp(2025, 4, 30)


class TestRandomTimestampSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> RandomTimestampSampler:
        return RandomTimestampSampler(dataset)

    def test_iter(self, sampler: RandomTimestampSampler) -> None:
        _, _, t = next(iter(sampler))
        assert TMIN <= t.start < t.stop <= TMAX

    def test_len(self, sampler: RandomTimestampSampler) -> None:
        assert len(sampler) == 3

    def test_toi(self, dataset: GeoDataset) -> None:
        tmin = pd.Timestamp(2025, 4, 10)
        tmax = pd.Timestamp(2025, 4, 20)
        toi = Interval(tmin, tmax)
        sampler = RandomTimestampSampler(dataset, toi=toi)
        _, _, t = next(iter(sampler))
        assert tmin <= t.start < t.stop <= tmax

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: RandomTimestampSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
