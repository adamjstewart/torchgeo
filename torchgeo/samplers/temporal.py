# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Temporal sampling routines."""

from collections.abc import Iterator

import pandas as pd
from pandas import Interval
from torch import Generator

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
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new TemporalSampler instance.

        Args:
            dataset: dataset to index from
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
            generator: Pseudo-random number generator (PRNG).
        """
        # TODO: do we want a length parameter?
        super().__init__(dataset, toi=toi)
        self.generator = generator

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

        for interval in intervals:
            x = y = slice(None)
            t = slice(interval.start, interval.stop)
            yield x, y, t
