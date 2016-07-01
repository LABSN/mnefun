# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
def anova_time(X):

    """A mass-univariate two-way ANOVA (with time as a co-variate)

    Parameters
    ----------
    X : array
        X should have the following dimensions:
            subjects x (2 conditions x N time points) x spatial locations
        This then calculates the paired t-values at each spatial location
        using time as a co-variate.

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
    from scipy import linalg, stats

    n_subjects, n_nested, n_sources = X.shape
    n_time = n_nested / 2
    # Turn Y into (2 x n_time x n_subjects) x n_sources
    Y = np.sqrt(np.reshape(X, (2 * n_time * n_subjects, n_sources), order='F'))
    cv, tv, sv = np.meshgrid(np.arange(2.0), np.arange(n_time),
                             np.arange(n_subjects), indexing='ij')
    dmat = patsy.dmatrix('C(cv) + C(tv) + C(sv)',
                         dict(sv=sv.ravel(), tv=tv.ravel(), cv=cv.ravel()))
    c = np.zeros((1, dmat.shape[1]))
    c[0, 1] = 1  # Contrast for just picking up condition difference
    b = np.dot(linalg.pinv(dmat), Y)
    d = Y - np.dot(dmat, b)
    r = dmat.shape[0] - np.linalg.matrix_rank(dmat)
    R = np.diag(np.dot(d.T, d))[:, np.newaxis] / r
    e = np.sqrt(R * np.dot(c, np.dot(linalg.pinv(np.dot(dmat.T, dmat)), c.T)))
    t = (np.dot(c, b) / e.T).T
    dof = r / (n_time - 1)  # Greenhouse-Geisser correction to the DOF
    p = np.sign(t) * 2 * stats.t.cdf(-abs(t), dof)
    return t, p, dof
