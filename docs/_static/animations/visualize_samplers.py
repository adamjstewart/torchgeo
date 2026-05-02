# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Visualize samplers."""

from geopandas import GeoSeries
from shapely import Polygon

from torchgeo.datasets import RasterDataset
from torchgeo.samplers import GridSpatialSampler, RandomSpatialSampler


class SpatialDataset(RasterDataset):
    """Fake spatial dataset."""

    def __init__(self) -> None:
        """Initialize a new SpatialDataset instance."""
        self.index = GeoSeries(
            [
                Polygon([(10, 0), (20, 10), (10, 20), (0, 10), (10, 0)]),
                Polygon([(20, 0), (30, 10), (20, 20), (10, 10), (20, 0)]),
            ]
        )
        self._res = (1, 1)


dataset = SpatialDataset()
samplers = [
    RandomSpatialSampler(dataset, size=3, generator=0),
    GridSpatialSampler(dataset, size=3, stride=2),
]
for sampler in samplers:
    ani = sampler.plot()
    ani.save(f'{sampler.__class__.__name__}.gif')
