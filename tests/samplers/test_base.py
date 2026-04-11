# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Iterator, Sequence

import pandas as pd
import pytest
import shapely
from _pytest.fixtures import SubRequest
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import Geometry
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import GeoSlice, Sample
from torchgeo.samplers import GeoSampler

MINT = pd.Timestamp(2025, 4, 24)
MAXT = pd.Timestamp(2025, 4, 25)


class CustomGeoDataset(GeoDataset):
    def __init__(
        self, geometry: Sequence[Geometry], res: tuple[float, float] = (10, 10)
    ) -> None:
        intervals = [(MINT, MAXT)] * len(geometry)
        index = pd.IntervalIndex.from_tuples(intervals, closed='both', name='datetime')
        crs = CRS.from_epsg(3005)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs=crs)
        self.res = res

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
            yield slice(i, i), slice(i, i), slice(MINT, MAXT)


class TestGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        geometry = [shapely.box(0, 0, 100, 100)]
        return CustomGeoDataset(geometry)

    @pytest.fixture(scope='class', params=[None, 5])
    def sampler(
        self, dataset: CustomGeoDataset, request: SubRequest
    ) -> CustomGeoSampler:
        return CustomGeoSampler(dataset, length=request.param)

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == (slice(0, 0), slice(0, 0), slice(MINT, MAXT))

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
