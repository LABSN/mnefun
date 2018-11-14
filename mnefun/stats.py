# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
from scipy import linalg, stats

import mne
from mne.stats import fdr_correction
from mne.externals.six import string_types


def anova_time(X, transform=True, signed_p=True, gg=True):
    """A mass-univariate two-way ANOVA (with time as a co-variate)

    Parameters
    ----------
    X : array
        X should have the following dimensions::

            (n_subjects, 2 * n_time, n_src)

        or::

            (n_subjects, 2, n_time, n_src)

        This then calculates the paired t-values at each spatial location
        using time as a co-variate.
    transform : bool
        If True, transform using the square root.
    signed_p : bool
        If True, change the p-value sign to match that of the t-statistic.
    gg : bool
        If True, correct DOF.

    Returns
    -------
    t : array
        t-values from the contrast, has the same length as the number of
        spatial locations.
    p : array
        Corresponding p values of the contrast.
    dof : int
        Degrees of freedom, with conservative Greenhouse-Geisser
        correction based on the number of time points (n_time - 1).
    """
    import patsy
    if X.ndim == 3:
        n_subjects, n_nested, n_src = X.shape
        n_time = n_nested // 2
        assert n_nested % 2 == 0
    else:
        assert X.ndim == 4
        n_subjects, n_cond, n_time, n_src = X.shape
        assert n_cond == 2
    X = np.reshape(X, (n_subjects, 2 * n_time, n_src))
    # Turn Y into (2 x n_time x n_subjects) x n_sources
    X = np.reshape(X, (2 * n_time * n_subjects, n_src), order='F')
    if transform:
        np.sqrt(X, out=X)
    cv, tv, sv = np.meshgrid(np.arange(2), np.arange(n_time),
                             np.arange(n_subjects), indexing='ij')
    dmat = patsy.dmatrix('C(cv) + C(tv) + C(sv)',
                         dict(cv=cv.ravel(), tv=tv.ravel(), sv=sv.ravel()))
    dof = dmat.shape[0] - np.linalg.matrix_rank(dmat)
    c = np.zeros((1, dmat.shape[1]))
    c[0, 1] = 1  # Contrast for just picking up condition difference
    # Equivalent, but slower here:
    assert np.isfinite(dmat).all()
    # b = np.dot(linalg.pinv(dmat), X)
    b = linalg.lstsq(dmat, X)[0]
    assert np.isfinite(b).all()
    X -= np.dot(dmat, b)
    X *= X
    R = np.sum(X, axis=0)[:, np.newaxis]
    R /= dof
    e = np.sqrt(R * np.dot(c, linalg.lstsq(np.dot(dmat.T, dmat), c.T)[0]))
    t = (np.dot(c, b) / e.T).T
    if n_time > 1 and gg:
        dof = dof / float(n_time - 1)  # Greenhouse-Geisser correction
    p = 2 * stats.t.cdf(-abs(t), dof)
    if signed_p:
        p *= np.sign(t)
    return t, p, dof


def hotelling_t2_baseline(stc, n_ave, baseline=(None, 0), check_baseline=True):
    """Compute p values from a baseline-corrected VectorSourceEstimate."""
    assert isinstance(stc, mne.VectorSourceEstimate)
    assert isinstance(baseline, tuple) and len(baseline) == 2
    mask = np.ones(len(stc.times), bool)
    if baseline[0] is not None:
        mask &= stc.times >= baseline[0]
    if baseline[1] is not None:
        mask &= stc.times <= baseline[1]
    assert mask.any()
    baseline = stc.data[..., mask]
    if check_baseline:
        np.testing.assert_allclose(baseline.mean(-1), 0.,
                                   atol=1e-6 * baseline.max())
    # Following definition 2 from:
    #
    #     https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
    #
    # Estimate the unbiased sample covariance matrix during the baseline:
    sigma = np.einsum('vot,vrt->vor', baseline, baseline)
    sigma /= (mask.sum() - 1)
    if check_baseline:
        np.testing.assert_allclose(np.cov(baseline[0]), sigma[0])
    # We need to correct this estimate by scaling by n_ave to get the value
    # that would have existed in the un-averaged data::
    #
    #     sigma *= n_ave
    #
    # But this cancels out with the sigma /= n_ave that we are supposed to do!
    # Now we compute the inverses of these covariance matrices:
    sinv = np.array([linalg.pinv(s, rcond=1e-6) for s in sigma])
    # And use these to compute the T**2 values:
    T2 = np.einsum('vot,von,vnt->vt', stc.data, sinv, stc.data)
    # Then convert T**2 for p variables and n DOF into F:
    #
    #     F_{p,n-p} = \frac{n-p}{p*(n-1)} * T ** 2
    #
    p = 3  # p
    F_dof = max(n_ave - p, 1)
    F = T2 * (F_dof / float(p * max(n_ave - 1, 1)))
    # compute associated p values
    p_val = 1 - stats.f.cdf(F, p, F_dof)
    assert p_val.shape == (stc.data.shape[0], stc.data.shape[2])
    stc = mne.SourceEstimate(
        p_val, stc.vertices, stc.tmin, stc.tstep, stc.subject, stc.verbose)
    return stc


def hotelling_t2(epochs, inv_op, src, baseline=(None, 0), update_interval=10):
    """Compute p values from a VectorSourceEstimate."""
    assert inv_op.ndim == 3 and inv_op.shape[1] == 3
    data = epochs.get_data()
    tmin, tstep = epochs.times[0], 1. / epochs.info['sfreq']
    del epochs
    n_ave = len(data)
    # Iterate over times
    F_dof = max(n_ave - 3, 1)
    # Following definition 2 from:
    #
    #     https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
    #
    F_p = np.zeros((inv_op.shape[0], data.shape[-1]))
    for ti in range(data.shape[-1]):
        if update_interval is not None and ti % update_interval == 0:
            print(' %s' % ti, end='')
        this_data = data[:, :, ti].T
        sens_cov = np.cov(this_data, ddof=1)
        # Compute the means
        mu = np.dot(inv_op, this_data.mean(-1))
        # Compute covariances and then invert them
        S_inv = np.einsum('vos,se,vre->vor', inv_op, sens_cov, inv_op)
        for ci, c in enumerate(S_inv):
            try:
                S_inv[ci] = linalg.inv(c)
            except np.linalg.LinAlgError:
                S_inv[ci] = linalg.pinv(c)
        # This is really a T**2, but to save mem we convert inplace
        F = np.einsum('vo,vor,vr->v', mu, S_inv, mu)
        F *= n_ave
        F *= F_dof / float(3 * max(n_ave - 1, 1))
        F_p[:, ti] = 1 - stats.f.cdf(F, 3, F_dof)
    stc = mne.SourceEstimate(
        F_p, [s['vertno'] for s in src], tmin, tstep, src[0]['subject_his_id'])
    return stc


def partial_conjunction(p, alpha=0.05, method='fisher', fdr_method='indep'):
    """Compute the partial conjunction map.

    Parameters
    ----------
    p : ndarray, shape (n_subjects, ...)
        The p-value maps for each subject at each location ``(...)``.
    alpha : float
        The FDR correction threshold for each conjunction map.
    method : str
        The method used to combine the p values, can be "fisher" (default)
        or "stouffer", both of which are valid for independent
        (across subjects) p-values only, and "simes", which should be
        suitable for dependent p-values.
    fdr_method : str
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli (across locations).

    Returns
    -------
    rejected : ndarray, shape (...)
        The largest number of subjects for which the null hypothesis
        was rejected at each location.
    p : ndarray, shape (n_subjects, ...)
        The group-corrected p-values for each subject count (first
        dimension) at each location.

    Notes
    -----
   Quoting [1]_:

        Let :math:`k` be the (unknown) number of conditions or subjects
        that show real effect. The problem of testing in every brain
        voxel :math:`v` whether at least :math:`u` out of :math:`n`
        conditions or subjects considered show real effects, can be
        generally stated as follows:

        .. math::

            H_{0v}^{u/n}: k<u\ \textrm{versus}\ H_{1v}^{u/n}: k \geq u

        We shall call :math:`H_{0v}^{u/n}` the partial conjunction
        null hypothesis.

    And note from [2]_:

        The (perhaps more) intuitive procedure in such settings, to apply
        an FDR controlling procedure on each p-value map separately and then
        take the intersection of the discovered locations, does not control
        the FDR of the combined discoveries.

    References
    ----------
    .. [1] Heller R, Golland Y, Malach R, Benjamini Y (2007).
       Conjunction group analysis: An alternative to mixed/random effect
       analysis. NeuroImage 37:1178–1185
       https://dx.doi.org/10.1016/j.neuroimage.2007.05.051
    .. [2] Benjamini Y, Heller R (2008). Screening for Partial Conjunction
       Hypotheses. Biometrics 64:1215–1222
       https://doi.org/10.1111/j.1541-0420.2007.00984.x
    """
    from scipy.stats import combine_pvalues
    known_types = ('fisher', 'stouffer', 'simes')
    if not isinstance(method, string_types) or method not in known_types:
        raise ValueError('Method must be one of %s, got %s'
                         % (known_types, method))
    p = np.array(p)  # copy
    bad = ((p <= 0) | (p > 1)).sum()
    if bad > 0:
        raise ValueError('All p-values must be positive and at most 1, got %s '
                         'invalid values' % (bad,))
    orig_shape = p.shape
    # Sort the p-values after effectively reshaping to (-1, n_subjects)
    p = np.reshape(p, (len(p), -1)).T
    p_sort = np.sort(p)
    # At each location
    for pi, pp in enumerate(p):
        # For each hypothesis count
        for ii in range(p.shape[1]):
            # combine the n - u + 1 largest p-values, and
            # when n=u this trivially collapses to the last (largest) value
            # (NB. ii = u - 1)
            # ii = 0 is the global null (should combine all p values), and
            # ii = p.shape[1] - 1 should use the maximum p value
            if method == 'simes':
                n = p.shape[1] - ii
                p[pi, ii] = n * (p_sort[pi, ii:] / np.arange(1, n + 1)).min()
            else:
                p[pi, ii] = combine_pvalues(p_sort[pi, ii:], method=method)[1]
    # FDR correct each map
    for ii in range(p.shape[1]):
        p[:, ii] = fdr_correction(p[:, ii], method=fdr_method)[1]
    # Get the subject count
    rejected = np.reshape((p < alpha).sum(-1), orig_shape[1:])
    p = np.reshape(p.T, orig_shape)
    return rejected, p
