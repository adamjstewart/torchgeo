# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Iterator

import pandas as pd
import pytest
import shapely
from _pytest.fixtures import SubRequest
from geopandas import GeoDataFrame, GeoSeries
from pandas import Interval
from pyproj import CRS
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import GeoSlice, Sample
from torchgeo.samplers import (
    GeoSampler,
    SpatialSampler,
    SpatioTemporalSampler,
    TemporalSampler,
)

TMIN = pd.Timestamp(2025, 4, 1)
TMAX = pd.Timestamp(2025, 4, 30)


class CustomGeoDataset(GeoDataset):
    def __init__(self) -> None:
        intervals = [(TMIN, TMAX)]
        index = pd.IntervalIndex.from_tuples(intervals, closed='both', name='datetime')
        geometry = [shapely.box(0, 0, 100, 100)]
        crs = CRS.from_epsg(3005)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs=crs)

    def __getitem__(self, index: GeoSlice) -> Sample:
        return {'bounds': self._slice_to_tensor(index)}


class CustomGeoSampler(GeoSampler):
    strategy = 'sequential'

    def __init__(self, dataset: GeoDataset, *, length: int | None = None) -> None:
        self.hidden_length = length or 5
        if length:
            self.length = length

    def __iter__(self) -> Iterator[GeoSlice]:
        for i in range(self.hidden_length):
            yield slice(i, i), slice(i, i), slice(TMIN, TMAX)


class CustomSpatialSampler(SpatialSampler):
    strategy = 'random'

    def __iter__(self) -> Iterator[GeoSlice]:
        series = GeoSeries([self.geometry])
        points = series.sample_points(size=5).explode()
        for point in points:
            yield slice(point.x, point.x), slice(point.y, point.y)


class CustomTemporalSampler(TemporalSampler):
    strategy = 'random'

    def _iter_subset(
        self, location: tuple[slice, slice] | None = None
    ) -> Iterator[GeoSlice]:
        intervals = self._init_subset(self.index, location)
        intervals = intervals.to_series().sample(frac=1)
        for interval in intervals:
            yield slice(None), slice(None), slice(interval.left, interval.right)


@pytest.fixture(scope='module')
def dataset() -> CustomGeoDataset:
    return CustomGeoDataset()


class TestGeoSampler:
    @pytest.fixture(scope='class', params=[None, 5])
    def sampler(
        self, dataset: CustomGeoDataset, request: SubRequest
    ) -> CustomGeoSampler:
        return CustomGeoSampler(dataset, length=request.param)

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == (slice(0, 0), slice(0, 0), slice(TMIN, TMAX))

    def test_len(self, sampler: CustomGeoSampler) -> None:
        assert len(sampler) == 5

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: CustomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestSpatialSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: CustomGeoDataset) -> CustomSpatialSampler:
        return CustomSpatialSampler(dataset)

    def test_iter(self, sampler: CustomSpatialSampler) -> None:
        x, y = next(iter(sampler))
        assert 0 <= x.start == x.stop <= 100
        assert 0 <= y.start == y.stop <= 100

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = shapely.box(0, 0, 10, 10)
        sampler = CustomSpatialSampler(dataset, roi=roi)
        x, y = next(iter(sampler))
        assert 0 <= x.start == x.stop <= 10
        assert 0 <= y.start == y.stop <= 10

    def test_len(self, sampler: CustomSpatialSampler) -> None:
        assert len(sampler) == 5

    def test_matmul(self, dataset: CustomGeoDataset) -> None:
        spatial_sampler = CustomSpatialSampler(dataset)
        temporal_sampler = CustomTemporalSampler(dataset)
        sampler = spatial_sampler @ temporal_sampler
        assert isinstance(sampler, SpatioTemporalSampler)

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SpatialSampler()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: CustomSpatialSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestTemporalSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: CustomGeoDataset) -> CustomTemporalSampler:
        return CustomTemporalSampler(dataset)

    def test_iter(self, sampler: CustomTemporalSampler) -> None:
        _, _, t = next(iter(sampler))
        assert t.start == TMIN and t.stop == TMAX

    def test_toi(self, dataset: CustomGeoDataset) -> None:
        tmin = pd.Timestamp(2025, 4, 10)
        tmax = pd.Timestamp(2025, 4, 20)
        toi = Interval(tmin, tmax)
        sampler = CustomTemporalSampler(dataset, toi=toi)
        _, _, t = next(iter(sampler))
        assert t.start == tmin and t.stop == tmax

    def test_subset(self, sampler: CustomTemporalSampler) -> None:
        x = y = slice(0, 10)
        _, _, t = next(iter(sampler._iter_subset((x, y))))
        assert t.start == TMIN and t.stop == TMAX

    def test_len(self, sampler: CustomTemporalSampler) -> None:
        assert len(sampler) == 1

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TemporalSampler()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self,
        dataset: CustomGeoDataset,
        sampler: CustomTemporalSampler,
        num_workers: int,
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
