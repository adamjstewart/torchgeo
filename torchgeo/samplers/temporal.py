# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Temporal sampling routines."""

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
from numpy.random import BitGenerator, Generator, RandomState, SeedSequence
from pandas import Interval, Timedelta

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice
from .base import TemporalSampler


class RandomTimestampSampler(TemporalSampler):
    """Sample individual timestamps from a time of interest randomly.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        toi: Interval | None = None,
        generator: int
        | BitGenerator
        | Generator
        | RandomState
        | SeedSequence
        | None = None,
    ) -> None:
        """Initialize a new RandomTimestampSampler instance.

        Args:
            dataset: Dataset to sample from.
            toi: Time of interest to sample from
                (defaults to the bounds of ``dataset.index``).
            generator: Pseudo-random number generator (PRNG).
        """
        # TODO: do we want a length parameter?
        super().__init__(dataset, toi=toi)
        self.generator = np.random.default_rng(generator)

    def _iter_subset(
        self, location: tuple[slice, slice] | None = None
    ) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Args:
            location: Region of interest to sample from.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
        # TODO: ensure we aren't modifying dataset.index too, may need to deepcopy
        index = self.index
        if location:
            # Since this only occurs in combination with a SpatialSampler, x and y are
            # guaranteed to have start and stop, and t is guaranteed to be empty
            x, y = location
            index = index.cx[x.start : x.stop, y.start : y.stop]

        intervals = index.index

        # Ensure time intervals are unique
        # Allows all intervals to be equally weighted, regardless of # locations
        intervals = pd.unique(intervals)

        intervals = intervals.sample(frac=1, random_state=self.generator)

        x = y = slice(None)
        for interval in intervals:
            t = slice(interval.start, interval.stop)
            yield x, y, t


class SequentialTimestampSampler(TemporalSampler):
    """Sample individual timestamps from a time of interest in order.

    .. versionadded:: 0.10
    """

    def _iter_subset(
        self, location: tuple[slice, slice] | None = None
    ) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Args:
            location: Region of interest to sample from.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
        # TODO: ensure we aren't modifying dataset.index too, may need to deepcopy
        index = self.index
        if location:
            # Since this only occurs in combination with a SpatialSampler, x and y are
            # guaranteed to have start and stop, and t is guaranteed to be empty
            x, y = location
            index = index.cx[x.start : x.stop, y.start : y.stop]

        intervals = index.index

        # Ensure time intervals are unique to avoid repeats
        intervals = sorted(pd.unique(intervals))

        x = y = slice(None)
        for interval in intervals:
            t = slice(interval.start, interval.stop)
            yield x, y, t


class RandomTimedeltaSampler(TemporalSampler):
    """Sample sliding window timedeltas from a time of interest randomly.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        delta: Timedelta,
        length: int | None = None,
        toi: Interval | None = None,
        generator: int
        | BitGenerator
        | Generator
        | RandomState
        | SeedSequence
        | None = None,
    ) -> None:
        """Initialize a new RandomTimedeltaSampler instance.

        Args:
            dataset: Dataset to sample from.
            delta: Duration of time.
            length: Number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                intervals that could be sampled from the dataset).
            toi: Time of interest to sample from
                (defaults to the bounds of ``dataset.index``).
            generator: Pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, toi=toi)
        self.delta = delta
        # TODO: should these be moved to _iter_subset? Length will change each epoch
        left = self.index.index.left.min()
        right = self.index.index.right.max()
        self.length = length or (right - left) // delta
        self.generator = np.random.default_rng(generator)

    def _iter_subset(
        self, location: tuple[slice, slice] | None = None
    ) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Args:
            location: Region of interest to sample from.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
        # TODO: ensure we aren't modifying dataset.index too, may need to deepcopy
        index = self.index
        if location:
            # Since this only occurs in combination with a SpatialSampler, x and y are
            # guaranteed to have start and stop, and t is guaranteed to be empty
            x, y = location
            index = index.cx[x.start : x.stop, y.start : y.stop]

        left = index.index.left.min()
        right = index.index.right.max() - self.delta

        i = 0
        x = y = slice(None)
        while i < self.length:
            tmin = self.generator.uniform(left, right)
            tmax = tmin + self.delta
            interval = Interval(tmin, tmax)
            if index.index.overlaps(interval):
                t = slice(interval.start, interval.stop)
                yield x, y, t
                i += 1


class SequentialTimedeltaSampler(TemporalSampler):
    """Sample sliding window timedeltas from a time of interest sequentially.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        delta: Timedelta,
        stride: Timedelta | None = None,
        toi: Interval | None = None,
    ) -> None:
        """Initialize a new SequentialTimedeltaSampler instance.

        Args:
            dataset: Dataset to sample from.
            delta: Duration of time.
            stride: Duration to skip between each sample (defaults to *delta*).
            toi: Time of interest to sample from
                (defaults to the bounds of ``dataset.index``).
        """
        super().__init__(dataset, toi=toi)
        self.delta = delta
        self.stride = stride or delta

    def _iter_subset(
        self, location: tuple[slice, slice] | None = None
    ) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Args:
            location: Region of interest to sample from.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
        # TODO: ensure we aren't modifying dataset.index too, may need to deepcopy
        index = self.index
        if location:
            # Since this only occurs in combination with a SpatialSampler, x and y are
            # guaranteed to have start and stop, and t is guaranteed to be empty
            x, y = location
            index = index.cx[x.start : x.stop, y.start : y.stop]

        left = index.index.left.min()
        right = index.index.right.max() - self.delta

        # TODO: make tile_to_chips more generic, support 1D inputs
        length = math.ceil((right - left - self.delta) / self.stride) + 1

        x = y = slice(None)
        for _ in range(length):
            interval = Interval(left, left + self.delta)
            if index.index.overlaps(interval):
                t = slice(interval.start, interval.stop)
                yield x, y, t
            left += self.delta
