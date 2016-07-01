# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import print_function
import numpy as np
from scipy import stats
from functools import partial
from mne.stats import ttest_1samp_no_p
from mne.stats import spatio_temporal_cluster_1samp_test as sct




def ttest_time(x, sigma=1e-3, n_perms=1024, init_thresh=0.05, tail=0,
               n_jobs=4):
    """1-sample t-test with permutation clustering in time for single source or
    sensor waveform

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
        If tail is 1, the statistic is thresholded above threshold.
        If tail is -1, the statistic is thresholded below threshold.
        If tail is 0, the statistic is thresholded on both sides of
        the distribution.
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).
"""
    if not isinstance(x, np.ndarray):
        raise TypeError("Input not array")
    thresh = -stats.distributions.t.ppf(init_thresh, x.shape[0] - 1)
    if tail == 0:
        thresh /= 2
    stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
    out = sct(x[:, :, np.newaxis], thresh, n_perms, tail, stat_fun,
              n_jobs=n_jobs, buffer_size=None, verbose=False)
    t_obs, clusters, cluster_pv, h0 = out
    print('    med:  %s' % np.median(t_obs.ravel()))
    # noinspection PyStringFormat
    print('    ps:   %s' % np.sort(cluster_pv)[:3])
    return t_obs, clusters, cluster_pv, h0
