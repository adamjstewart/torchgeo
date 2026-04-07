# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Spatial sampling routines."""

import math
from collections.abc import Iterator

import numpy as np
import shapely
from geopandas import GeoSeries
from numpy.random import BitGenerator, Generator, SeedSequence
from shapely import Polygon

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice
from .base import SpatialSampler
from .constants import Units
from .utils import _to_tuple, tile_to_chips


class RandomSpatialSampler(SpatialSampler):
    """Sample locations from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    .. versionadded:: 0.10
    """

    strategy = 'random'

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: Polygon | None = None,
        units: Units = Units.PIXELS,
        generator: int | BitGenerator | Generator | SeedSequence | None = None,
    ) -> None:
        """Initialize a new RandomSpatialSampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: Dataset to sample from.
            size: Dimensions of each :term:`patch`.
            length: Number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset).
            roi: Region of interest to sample from
                (defaults to the bounds of ``dataset.index``).
            units: Defines if ``size`` is in pixel or CRS units.
            generator: Pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, roi=roi)

        self.size = _to_tuple(size)
        self.generator = np.random.default_rng(generator)

        # Convert from pixel units to CRS units
        if units == Units.PIXELS:
            self.size = (self.size[0] * dataset.res[1], self.size[1] * dataset.res[0])

        # Erosion to avoid out-of-bounds sampling
        # Purposefully conservative radius calculation
        # TODO: this operation removes Point and LineString, should we keep these?
        distance = math.sqrt((self.size[0] / 2) ** 2 + (self.size[1] / 2) ** 2)
        self.geometry = shapely.buffer(self.geometry, -distance)

        # Default to approximate number of non-overlapping patches
        patch_area = self.size[0] * self.size[1]
        self.length = length or shapely.area(self.geometry) // patch_area

    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """
        # Ensure a new set of random points for each epoch
        series = GeoSeries([self.geometry])
        points = series.sample_points(size=self.length, rng=self.generator)

        for point in points:
            # TODO: snap to pixel grid? How? Can use outer geometry, but not file-specific, users will have to use TAP more
            # IDEA: snap to pixel grid in RasterDataset, not here. Or maybe that would be incredibly confusing...
            xmin = point.x - self.size[1] / 2
            xmax = point.x + self.size[1] / 2
            ymin = point.y - self.size[0] / 2
            ymax = point.y + self.size[0] / 2
            yield slice(xmin, xmax), slice(ymin, ymax)

    def __len__(self) -> int:
        """Length of each epoch."""
        return self.length


class GridSpatialSampler(SpatialSampler):
    """Sample locations from a region of interest in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``size - stride``) should be approximately equal to
    the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of the
    CNN.

    .. versionadded:: 0.10
    """

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
        """Initialize a new GridSpatialSampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: Dataset to sample from.
            size: Dimensions of each :term:`patch`.
            stride: Distance to skip between each patch (defaults to *size*).
            roi: Region of interest to sample from
                (defaults to the bounds of ``dataset.index``).
            units: Defines if ``size`` and ``stride`` are in pixel or CRS units.
        """
        super().__init__(dataset, roi=roi)

        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride or size)

        # Convert from pixel units to CRS units
        if units == Units.PIXELS:
            self.size = (self.size[0] * dataset.res[1], self.size[1] * dataset.res[0])
            self.stride = (
                self.stride[0] * dataset.res[1],
                self.stride[1] * dataset.res[0],
            )

    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """
        bounds = self.geometry.bounds

        # TODO: adjust xmin/ymin to have equal spacing in case of non-integer multiple
        # xmid, ymid = self.geometry.centroid

        rows, cols = tile_to_chips(bounds, self.size, self.stride)

        # For each row...
        for i in range(rows):
            ymin = bounds[1] + i * self.stride[0]
            ymax = ymin + self.size[0]

            # For each column...
            for j in range(cols):
                xmin = bounds[0] + j * self.stride[1]
                xmax = xmin + self.size[1]

                # Check for intersection
                bbox = shapely.box(xmin, ymin, xmax, ymax)
                if self.geometry.intersects(bbox):
                    yield slice(xmin, xmax), slice(ymin, ymax)
