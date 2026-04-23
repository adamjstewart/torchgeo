# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo sampler base classes."""

import abc
from abc import ABC
from collections.abc import Iterator
from typing import Literal

import numpy as np
from pandas import Interval, IntervalIndex
from shapely import Polygon
from torch.utils.data import Sampler

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice


# TODO: alternatively call it BaseSampler to have more backwards-compatibility when deprecating the prior GeoSampler
class GeoSampler(Sampler[GeoSlice], ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns a GeoSlice that can uniquely index any :class:`~torchgeo.datasets.GeoDataset`.
    """

    length: int

    @property
    @abc.abstractmethod
    def strategy(self) -> Literal['random', 'sequential']:
        """Sampling strategy.

        All sampling strategies can be categorized as either being random or sequential.
        This distinction only matters when combining samplers via
        :class:`SpatioTemporalSampler`, where either a zip (random) or product
        (sequential) of all sample locations is taken during each epoch.

        Returns:
            One of 'random' or 'sequential'.
        """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """

    def __len__(self) -> int:
        """Length of each epoch.

        Returns:
            The sampler length.
        """
        if hasattr(self, 'length'):
            # If length is known, use it
            return self.length
        else:
            # Otherwise, use brute force
            return sum(1 for _ in self)


class SpatialSampler(GeoSampler):
    """Abstract base class for all spatial sampling strategies.

    .. versionadded:: 0.10
    """

    def __init__(self, dataset: GeoDataset, *, roi: Polygon | None = None) -> None:
        """Initialize a new SpatialSampler instance.

        Args:
            dataset: Dataset to sample from.
            roi: Region of interest to sample from
                (defaults to the bounds of ``dataset.index``).
        """
        # Create one single MultiPolygon of all objects
        # Allows all locations to be equally weighted, regardless of # time stamps
        # TODO: ensure we aren't modifying dataset.index too, may need to deepcopy
        self.geometry = dataset.index.geometry.union_all()

        if roi is not None:
            self.geometry &= roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """

    def __matmul__(self, other: 'TemporalSampler') -> 'SpatioTemporalSampler':
        """Compute the product of two samplers.

        Args:
            other: A temporal sampling strategy.

        Returns:
            A single spatial and temporal sampler.
        """
        return SpatioTemporalSampler(self, other)


class TemporalSampler(GeoSampler):
    """Abstract base class for all temporal sampling strategies.

    .. versionadded:: 0.10
    """

    def __init__(self, dataset: GeoDataset, *, toi: Interval | None = None) -> None:
        """Initialize a new TemporalSampler instance.

        Args:
            dataset: Dataset to sample from.
            toi: Time of interest to sample from
                (defaults to the bounds of ``dataset.index``).
        """
        self.index = dataset.index

        if toi:
            tmin = np.maximum(toi.left.to_datetime64(), self.index.index.left)
            tmax = np.minimum(toi.right.to_datetime64(), self.index.index.right)
            valid = tmax >= tmin
            tmin = tmin[valid]
            tmax = tmax[valid]
            self.index = self.index[valid]
            self.index.index = IntervalIndex.from_arrays(
                tmin, tmax, closed='both', name='datetime'
            )

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
        yield from self._iter_subset()

    def _init_subset(
        self, location: tuple[slice, slice] = (slice(None), slice(None))
    ) -> IntervalIndex:
        """Narrow down index to a specific location.

        Args:
            location: A specific location.

        Returns:
            A subset of *self.index* at *location*.
        """
        index = self.index

        # TODO: ensure we aren't modifying dataset.index too, may need to deepcopy
        if location:
            # Since this only occurs in combination with a SpatialSampler, x and y are
            # guaranteed to have start and stop, and t is guaranteed to be empty
            x, y = location
            index = index.cx[x.start : x.stop, y.start : y.stop]

        return index.index  # ty: ignore[invalid-return-type]

    @abc.abstractmethod
    def _iter_subset(
        self, location: tuple[slice, slice] = (slice(None), slice(None))
    ) -> Iterator[tuple[slice, slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Args:
            location: Region of interest to sample from.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """


class SpatioTemporalSampler(GeoSampler):
    """Product of a spatial and a temporal sampler.

    .. versionadded:: 0.10
    """

    # akin to IntersectionDataset
    # name: ZipSampler? ProductSampler?

    # TODO: this parameter will be ignored, maybe move abc to spatial/temporal?
    strategy = 'random'

    def __init__(
        self, spatial_sampler: SpatialSampler, temporal_sampler: TemporalSampler
    ) -> None:
        """Initialize a new SpatioTemporalSampler instance.

        Args:
            spatial_sampler: A spatial sampling strategy.
            temporal_sampler: A temporal sampling strategy.
        """
        self.spatial_sampler = spatial_sampler
        self.temporal_sampler = temporal_sampler

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        match self.spatial_sampler.strategy, self.temporal_sampler.strategy:
            case 'random', 'random':
                for _ in range(len(self.spatial_sampler) * len(self.temporal_sampler)):
                    location = next(iter(self.spatial_sampler))
                    yield next(iter(self.temporal_sampler._iter_subset(location)))
            case 'sequential', 'sequential':
                for location in iter(self.spatial_sampler):
                    for index in iter(self.temporal_sampler._iter_subset(location)):
                        yield index
            case 'random', 'sequential':
                for _ in range(len(self.spatial_sampler)):
                    location = next(iter(self.spatial_sampler))
                    for index in iter(self.temporal_sampler._iter_subset(location)):
                        yield index
            case 'sequential', 'random':
                for location in iter(self.spatial_sampler):
                    for _ in range(len(self.temporal_sampler)):
                        yield next(iter(self.temporal_sampler._iter_subset(location)))
