# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import print_function

import numpy as np
import scipy.stats as stats
from mne import grand_average


def grandaverage_evokeds(evokeds):
    """helper for creating group averaged evoked object. See mne.grand_average.

        Parameters
        ----------
        evokeds : list
            List of mne evoked objects
    """
    return grand_average(evokeds)


def numpy_weighted_mean(data, alpha=.95, weights=None):
    """Calculate the weighted mean of an array/list using numpy.

        Parameters
        ----------
        data : array
            Input time series data of shape subjects x time.
        alpha : float
            Default to .95 to return 95% confidence interval about mean.
        Returns
        -------
        mean : array
        sem : array
        hyp : float
    """
    sem = stats.sem(data)
    hyp = sem * stats.t._ppf((1 + (1 - alpha)) / 2., data.shape[0] - 1)
    if weights is None:
        mean = np.mean(data, axis=0)
    else:
        weights = np.array(weights).flatten() / float(sum(weights))
        mean = np.dot(np.array(data), weights)
    return mean, sem, hyp


def numpy_weighted_median(data, weights=None):
    """Calculate the weighted median of an array/list using numpy.

    Parameters
    ----------
    x : array
        contrast array with dimensions subjects x time

    Notes
    -----
    https://github.com/tinybike/weightedstats
    """
    if weights is None:
        return np.median(np.array(data).flatten())
    data, weights = np.array(data).flatten(), np.array(weights).flatten()
    if any(weights > 0):
        sorted_data, sorted_weights = map(np.array,
                                          zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(sorted_weights)
        if any(weights > midpoint):
            return (data[weights == np.max(weights)])[0]
        cumulative_weight = np.cumsum(sorted_weights)
        below_midpoint_index = np.where(cumulative_weight <= midpoint)[0][-1]
        if cumulative_weight[
            below_midpoint_index] - midpoint < sys.float_info.epsilon:
            return np.mean(
                sorted_data[below_midpoint_index:below_midpoint_index + 2])
        return sorted_data[below_midpoint_index + 1]


