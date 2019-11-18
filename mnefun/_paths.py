# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import os
from os import path as op
import re
import numpy as np

from mne.io import Raw


def safe_inserter(string, inserter):
    """Helper to insert a subject name into a string if %s is present

    Parameters
    ----------
    string : str
        String to fill.

    inserter : str
        The string to put in ``string`` if ``'%s'`` is present.

    Returns
    -------
    string : str
        The modified string.
    """
    if '%s' in string:
        string = string % inserter
    return string


def get_event_fnames(p, subj, run_indices=None):
    """Get event filenames for a subject

    Parameters
    ----------
    p : instance of Params
        Parameters structure.
    subj : str
        Subject name.
    run_indices : array-like | None
        The run indices to include. None will include all.

    Returns
    -------
    fnames : list
        List of filenames.

    Notes
    -----
    This function will create the list file output directory if it does
    not exist.
    """
    if run_indices is None:
        run_indices = np.arange(len(p.run_names))
    run_names = [r for ri, r in enumerate(p.run_names) if ri in run_indices]

    lst_dir = op.join(p.work_dir, subj, p.list_dir)
    if not op.isdir(lst_dir):
        os.mkdir(lst_dir)

    fnames = [op.join(lst_dir, 'ALL_' + safe_inserter(r, subj) + '-eve.lst')
              for r in run_names]
    return fnames


def _regex_convert(f):
    """Helper to regex a given filename (for split file purposes)"""
    return '.*%s-?[0-9]*.fif$' % op.basename(f)[:-4]


def get_raw_fnames(p, subj, which='raw', erm=True, add_splits=False,
                   run_indices=None):
    """Get raw filenames

    Parameters
    ----------
    p : instance of Params
        Parameters structure.
    subj : str
        Subject name.
    which : str
        Type of raw filenames. Must be 'sss', 'raw', or 'pca'.
    erm : bool | str
        If True, include empty-room files (appended to end). If 'only', then
        only return empty-room files.
    add_splits : bool
        If True, add split filenames if they exist. This should only
        be necessary for Maxfilter-related things. Will only return files
        that actually already exist.
    run_indices : array-like | None
        The run indices to include. None will include all.

    Returns
    -------
    fnames : list
        List of filenames.
    """
    assert which in ('sss', 'raw', 'pca')
    if which == 'sss':
        raw_dir = op.join(p.work_dir, subj, p.sss_dir)
        tag = p.sss_fif_tag
    elif which == 'raw':
        raw_dir = op.join(p.work_dir, subj, p.raw_dir)
        tag = p.raw_fif_tag
    elif which == 'pca':
        raw_dir = op.join(p.work_dir, subj, p.pca_dir)
        tag = p.pca_extra + p.sss_fif_tag

    if run_indices is None:
        run_indices = np.arange(len(p.run_names))
    run_names = [r for ri, r in enumerate(p.run_names) if ri in run_indices]

    if erm == 'only':
        use = p.runs_empty
    elif erm:
        use = run_names + p.runs_empty
    else:
        use = run_names
    fnames = [safe_inserter(r, subj) + tag for r in use]
    if add_splits:
        regexs = [re.compile(_regex_convert(f)) for f in fnames]
        fnames = sorted([op.join(raw_dir, f) for f in os.listdir(raw_dir)
                         if any(r.match(f) is not None for r in regexs)])
    fnames = [op.join(raw_dir, f) for f in fnames]
    return fnames


def get_cov_fwd_inv_fnames(p, subj, run_indices):
    """Get covariance, forward, and inverse filenames for a subject"""
    cov_fnames = []
    fwd_fnames = []
    inv_fnames = []
    inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
    fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
    cov_dir = op.join(p.work_dir, subj, p.cov_dir)
    make_erm_inv = len(p.runs_empty) > 0

    # Shouldn't matter which raw file we use
    raw_fname = get_raw_fnames(p, subj, 'pca', True, False, run_indices)[0]
    if op.isfile(raw_fname):
        raw = Raw(raw_fname)
        meg, eeg = 'meg' in raw, 'eeg' in raw
    else:
        meg = eeg = True

    out_flags, meg_bools, eeg_bools = [], [], []
    if meg:
        out_flags += ['-meg']
        meg_bools += [True]
        eeg_bools += [False]
    if eeg:
        out_flags += ['-eeg']
        meg_bools += [False]
        eeg_bools += [True]
    if meg and eeg:
        out_flags += ['-meg-eeg']
        meg_bools += [True]
        eeg_bools += [True]
    if make_erm_inv:
        cov_fnames += [op.join(cov_dir, safe_inserter(p.runs_empty[0], subj) +
                               p.pca_extra + p.inv_tag + '-cov.fif')]
    for name in p.inv_names:
        s_name = safe_inserter(name, subj)
        temp_name = s_name + ('-%d' % p.lp_cut) + p.inv_tag
        fwd_fnames += [op.join(fwd_dir, s_name + p.inv_tag + '-fwd.fif')]
        cov_fnames += [op.join(cov_dir, safe_inserter(name, subj) +
                               ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')]
        for f, m, e in zip(out_flags, meg_bools, eeg_bools):
            for l, s, x in zip([None, 0.2], [p.inv_fixed_tag, ''],
                               [True, False]):
                inv_fnames += [op.join(inv_dir,
                                       temp_name + f + s + '-inv.fif')]
                if (not e) and make_erm_inv:
                    inv_fnames += [op.join(inv_dir, temp_name + f +
                                           p.inv_erm_tag + s + '-inv.fif')]
    return cov_fnames, fwd_fnames, inv_fnames


def get_epochs_evokeds_fnames(p, subj, analyses, remove_unsaved=False):
    """Get epochs and evoked filenames for a subject"""
    epochs_dir = op.join(p.work_dir, subj, p.epochs_dir)
    evoked_dir = op.join(p.work_dir, subj, p.inverse_dir)
    mat_file = op.join(epochs_dir, '%s_%d' % (p.epochs_prefix, p.lp_cut) +
                       p.inv_tag + '_' + subj + p.epochs_tag + '.mat')
    fif_file = op.join(epochs_dir, '%s_%d' % (p.epochs_prefix, p.lp_cut) +
                       p.inv_tag + '_' + subj + p.epochs_tag + '.fif')
    epochs_fnames = [fname
                     for fname, c in zip([mat_file, fif_file], ['mat', 'fif'])
                     if not remove_unsaved or c in p.epochs_type]

    evoked_fnames = []
    for analysis in analyses:
        fn = '%s_%d%s_%s_%s-ave.fif' % (analysis, p.lp_cut, p.inv_tag,
                                        p.eq_tag, subj)
        evoked_fnames.append(op.join(evoked_dir, fn))

    return epochs_fnames, evoked_fnames


def get_report_fnames(p, subj):
    """Get filenames of report files."""
    fnames = [op.join(p.work_dir, subj, '%s_fil%d_report.html'
                      % (subj, p.lp_cut))]
    return fnames


def get_proj_fnames(p, subj):
    """Get filenames of projections files."""
    proj_fnames = []
    proj_dir = op.join(p.work_dir, subj, p.pca_dir)
    for fn in ['preproc_all-proj.fif', 'preproc_ecg-proj.fif',
               'preproc_blink-proj.fif', 'preproc_cont-proj.fif']:
        if op.isfile(op.join(proj_dir, fn)):
            proj_fnames.append(fn)
    return proj_fnames


def get_bad_fname(p, subj, check_exists=True):
    """Get filename for post-SSS bad channels."""
    bad_dir = op.join(p.work_dir, subj, p.bad_dir)
    if not op.isdir(bad_dir):
        os.mkdir(bad_dir)
    bad_file = op.join(bad_dir, 'bad_ch_' + subj + p.bad_tag)
    if check_exists:
        bad_file = None if not op.isfile(bad_file) else bad_file
    return bad_file
