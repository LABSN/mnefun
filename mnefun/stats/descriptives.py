# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import print_function

from os import path as op
import numpy as np
import scipy.stats as stats
from functools import partial
from mne import (read_evokeds, grand_average)
from mne.stats import ttest_1samp_no_p
from mne.stats import spatio_temporal_cluster_1samp_test as sct


def gravrevokeds(directory, subjects, analysis, condition, filtering,
                 baseline=(None, 0)):
    """helper for creating group averaged evoked file

    Parameters
    ----------
    directory : str
        MNEFUN parent study database directory.
    subjects : list
        List of subjects to combine evoked data across.
    analysis : str
        Evoked data set name.
    condition : str
        Evoked condition.
    filtering : int
        Low pass filter setting used to create evoked files.
    baseline : None | tuple
        The time interval to apply baseline correction.
        If None do not apply it.
        If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None
        then b is set to the end of the interval. If baseline is equal
        to (None, None) all the time interval is used.
    """
    evokeds = []
    for subj in subjects:
        evoked_file = op.join(directory, subj, 'inverse',
                              '%s_%d-sss_eq_%s-ave.fif' % (analysis, filtering,
                                                           subj))
        evokeds.append(read_evokeds(evoked_file, condition=condition,
                                    baseline=baseline))
    grandavr = grand_average(evokeds)
    return grandavr


def numpy_weighted_mean(data, alpha=.05, weights=None):
    """Calculate the weighted mean of an array/list using numpy.

    Parameters
    ----------
    x : array
        contrast array with dimensions subjects x time
    alpha : float
        Default to .05 to return 95% confidence interval about mean.

    Returns
    -------
    mean_ts : array type
    se_ts : array type
    hyp : float
    """
    sem = sp.stats.sem(data)
    hyp = se * stats.t._ppf((1 + (1 - alpha)) / 2., n - 1)
    if weights is None:
        mean = np.mean(data, axis=0)
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
        if cumulative_weight[below_midpoint_index] - midpoint < sys.float_info.epsilon:
            return np.mean(sorted_data[below_midpoint_index:below_midpoint_index+2])
        return sorted_data[below_midpoint_index+1]


def ttest_time(x, sigma=1e-3, n_perms=1024, init_thresh=0.05, tail=0,
                   n_jobs=4):
    """1-sample t-test with permutation clustering in time

    Parameters
    ----------
    x : array
        contrast array with dimensions subjects x time
    sigma : float
    n_perms : int
        The number of permutations to compute.
    init_thresh : float
    tail : int
        -1 or 0 or 1 (default = 0)
        If tail is 1, the statistic is thresholded above threshold. If tail is -1,
        the statistic is thresholded below threshold. If tail is 0, the statistic is
        thresholded on both sides of the distribution.
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).
    """
    thresh = -stats.distributions.t.ppf(init_thresh, x.shape[0] - 1)
    if tail == 0:
        thresh /= 2
    stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
    stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
    out = sct(x, thresh, n_perms, tail, stat_fun, n_jobs=n_jobs, buffer_size=None, verbose=False)
    t_obs, clusters, cluster_pv, h0 = out
    print('    med:  %s' % np.median(t_obs.ravel()))
    # noinspection PyStringFormat
    print('    ps:   %s' % np.sort(cluster_pv)[:3])
    return t_obs, clusters, cluster_pv, h0
