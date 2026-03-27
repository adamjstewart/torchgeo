# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Temporal sampling routines."""

from collections.abc import Iterator

from pandas import Interval

from ..datasets import GeoDataset
from ..datasets.utils import GeoSlice
from .base import TemporalSampler


class RandomTemporalSampler(TemporalSampler):
    """Samples elements from a time of interest randomly.

    .. versionadded:: 0.10
    """

    def __init__(self, dataset: GeoDataset, *, toi: Interval | None = None) -> None:
        """Initialize a new TemporalSampler instance.

        Args:
            dataset: dataset to index from
            toi: time of interest to sample from
                (defaults to the bounds of ``dataset.index``)
        """
        # what else do we need?
        # size? in what units?

    def _iter_subset(self, index: GeoSlice | None = None) -> Iterator[GeoSlice]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [:, :, tmin:tmax] coordinates to index a dataset.
        """
