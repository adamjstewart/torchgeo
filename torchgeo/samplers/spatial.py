# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Spatial sampling routines."""

import math
from collections.abc import Iterator

import shapely
from geopandas import GeoSeries
from shapely import Polygon
from torch import Generator

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice
from .base import SpatialSampler
from .constants import Units
from .utils import _to_tuple


class RandomSpatialSampler(SpatialSampler):
    """Random spatial sampling."""

    strategy = 'random'

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: Polygon | None = None,
        units: Units = Units.PIXELS,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new RandomSpatialSampler instance."""
        super().__init__(dataset, roi)
        # TODO: units
        self.size = _to_tuple(size)

        # Erosion to avoid out-of-bounds sampling
        # Purposefully conservative radius calculation
        # TODO: this operation removes Point and LineString, should we keep these?
        distance = math.sqrt((self.size[0] / 2) ** 2 + (self.size[1] / 2) ** 2)
        self.geometry = shapely.buffer(self.geometry, -distance)

        # Default to approximate number of non-overlapping patches
        patch_area = self.size[0] * self.size[1]
        self.length = length or shapely.area(self.geometry) // patch_area

    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over the sampler."""
        # Ensure a new set of random points for each epoch
        series = GeoSeries([self.geometry])
        points = series.sample_points(size=self.length)

        for point in points:
            # TODO: snap to pixel grid? How? Can use outer geometry, but not file-specific, users will have to use TAP more
            x = slice(point.x - self.size[1] / 2, point.x + self.size[1] / 2)
            y = slice(point.y - self.size[0] / 2, point.y + self.size[0] / 2)
            yield x, y

        # precompute and then yield from?

    def __len__(self) -> int:
        """Length of each epoch."""
        return self.length


class GridSpatialSampler(SpatialSampler):
    """Gridded spatial sampling."""

    strategy = 'sequential'

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        size: tuple[float, float] | float,
        stride: tuple[float, float] | float | None = None,
        roi: Polygon | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new GridSpatialSampler instance."""

    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over the sampler."""

        # precompute and then yield from?
