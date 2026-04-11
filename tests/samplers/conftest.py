# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import pytest
import shapely
from geopandas import GeoDataFrame
from pyproj import CRS

from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import GeoSlice, Sample


class CustomGeoDataset(GeoDataset):
    def __init__(self) -> None:
        intervals = [
            (pd.Timestamp(2025, 4, 1), pd.Timestamp(2025, 4, 2)),
            (pd.Timestamp(2025, 4, 15), pd.Timestamp(2025, 4, 16)),
            (pd.Timestamp(2025, 4, 29), pd.Timestamp(2025, 4, 30)),
        ]
        index = pd.IntervalIndex.from_tuples(intervals, closed='both', name='datetime')
        geometry = [
            shapely.box(0, 0, 100, 100),
            shapely.box(0, 0, 10, 10),
            shapely.box(90, 90, 100, 100),
        ]
        crs = CRS.from_epsg(3005)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs=crs)
        self.res = 2

    def __getitem__(self, index: GeoSlice) -> Sample:
        return {'bounds': self._slice_to_tensor(index)}


@pytest.fixture(scope='package')
def dataset() -> GeoDataset:
    return CustomGeoDataset()
