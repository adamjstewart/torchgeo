# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import math

import pytest
import shapely
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.samplers import GridSpatialSampler, RandomSpatialSampler, Units


class TestRandomSpatialSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> RandomSpatialSampler:
        return RandomSpatialSampler(dataset, size=5)

    def test_iter(self, sampler: RandomSpatialSampler) -> None:
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 100
        assert 0 <= y.start < y.stop <= 100
        assert math.isclose(x.stop - x.start, 10)
        assert math.isclose(y.stop - y.start, 10)

    def test_len(self, sampler: RandomSpatialSampler) -> None:
        assert len(sampler) == 100

    def test_length(self, dataset: GeoDataset) -> None:
        sampler = RandomSpatialSampler(dataset, size=5, length=99)
        assert len(sampler) == 99

    def test_roi(self, dataset: GeoDataset) -> None:
        roi = shapely.box(0, 0, 20, 20)
        sampler = RandomSpatialSampler(dataset, size=5, roi=roi)
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 20
        assert 0 <= y.start < y.stop <= 20

    def test_units(self, dataset: GeoDataset) -> None:
        sampler = RandomSpatialSampler(dataset, size=10, units=Units.CRS)
        assert len(sampler) == 100

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: RandomSpatialSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestGridSpatialSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> GridSpatialSampler:
        return GridSpatialSampler(dataset, size=5)

    def test_iter(self, sampler: GridSpatialSampler) -> None:
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 100
        assert 0 <= y.start < y.stop <= 100
        assert math.isclose(x.stop - x.start, 10)
        assert math.isclose(y.stop - y.start, 10)

    def test_len(self, sampler: GridSpatialSampler) -> None:
        assert len(sampler) == 100

    def test_stride(self, dataset: GeoDataset) -> None:
        sampler = GridSpatialSampler(dataset, size=5, stride=2.5)
        assert len(sampler) == 361

    def test_roi(self, dataset: GeoDataset) -> None:
        roi = shapely.box(0, 0, 20, 20)
        sampler = GridSpatialSampler(dataset, size=5, roi=roi)
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 20
        assert 0 <= y.start < y.stop <= 20

    def test_units(self, dataset: GeoDataset) -> None:
        sampler = GridSpatialSampler(dataset, size=10, units=Units.CRS)
        assert len(sampler) == 100

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: GridSpatialSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
