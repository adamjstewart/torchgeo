# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo sampler base classes."""

import abc
from abc import ABC
from collections.abc import Iterator
from typing import Literal

import pandas as pd
from pandas import Interval
from shapely import Polygon
from torch.utils.data import Sampler

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice


class GeoSampler(Sampler[GeoSlice], ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns a GeoSlice that can uniquely index any :class:`~torchgeo.datasets.GeoDataset`.
    """

    @abc.abstractmethod
    @property
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


class SpatialSampler(GeoSampler):
    """Abstract base class for all spatial sampling strategies."""

    def __init__(self, dataset: GeoDataset, roi: Polygon | None = None) -> None:
        """Initialize a new SpatialSampler instance.

        Args:
            dataset: Dataset to index from.
            roi: Region of interest to sample from
                (defaults to the bounds of ``dataset.index``).
        """
        # Create one single MultiPolygon of all bounds
        # Allows all locations to be equally weighted, regardless of # time stamps
        # TODO: ensure we aren't modifying dataset.index too, may need to deepcopy
        self.geometry = dataset.index.geometry.union_all()

        if roi:
            self.geometry &= roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[GeoSlice]:
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
    """Abstract base class for all temporal sampling strategies."""

    def __init__(self, dataset: GeoDataset, toi: Interval | None = None) -> None:
        """Initialize a new TemporalSampler instance.

        Args:
            dataset: dataset to index from
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
        """
        # Ensure all times are unique
        self.index = pd.unique(dataset.index)

        if toi:
            self.index = self.index.iloc[self.index.index.overlaps(toi)]

    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
        yield from self._iter_subset()

    @abc.abstractmethod
    def _iter_subset(self, index: GeoSlice | None) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """


class SpatioTemporalSampler(GeoSampler):
    """Product of a spatial and a temporal sampler."""

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

    def __iter__(self) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax, tmin:tmax] coordinates to index a dataset.
        """
        match self.spatial_sampler.strategy, self.temporal_sampler.strategy:
            case 'random', 'random':
                for _ in range(len(self)):
                    location = next(iter(self.spatial_sampler))
                    index = next(iter(self.temporal_sampler._subset_iter(location)))
                    yield index
            case 'sequential', 'sequential':
                for location in self.spatial_sampler:
                    for index in self.temporal_sampler._subset_iter(location):
                        yield index
            # TODO: random-sequential, sequential-random

    def __len__(self) -> int:
        """Length of each epoch.

        Returns:
            The sampler length.
        """
        # Or min? max?
        # Should this depend on random or sequential?
        # Is this even possible to know ahead of time?
        return len(self.spatial_sampler) * len(self.temporal_sampler)
