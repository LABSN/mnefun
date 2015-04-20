# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from os import path as op
import glob
import numpy as np
import re
import warnings
from shutil import move

from mne.io import Raw
from mne import pick_types

from ._paths import get_raw_fnames


def fix_eeg_channels(raw_files, anon=None, verbose=True):
    """Reorder EEG channels based on UW cap setup

    Parameters
    ----------
    raw_files : list of str | str
        The raw file name(s) to reorder, if it has not been done yet.
    anon : dict | None
        If None, no anonymization is done. If dict, should have the following:
        ``['first_name', 'last_name', 'birthday']``. Names should be strings,
        while birthday should be a tuple of ints (year, month, day).
    verbose : bool
        If True, print whether or not the files were modified.
    """
    order = np.array([1, 2, 3, 5, 6, 7, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 30,
                      31, 32, 33, 34, 35, 36, 37, 38,
                      41, 42, 43, 44, 45, 46, 47, 48, 49,
                      51, 52, 54, 55, 56, 57, 58, 60,
                      39, 29, 18, 4, 8, 28, 40, 59, 50, 53]) - 1
    assert len(order) == 60
    write_key = 'LABSN_EEG_REORDER:' + ','.join([str(o) for o in order])
    if anon is None:
        anon_key = ''
    else:
        anon_key = ';anonymized'

    # do some type checking
    if not isinstance(raw_files, list):
        raw_files = [raw_files]

    # actually do the reordering
    for ri, raw_file in enumerate(raw_files):
        need_reorder, need_anon, write_key, anon_key, picks, order = \
            _is_file_unfixed(raw_file, anon)
        if need_anon or need_reorder:
            to_do = []
            if need_reorder:
                to_do += ['reordering']
            if need_anon:
                to_do += ['anonymizing']
            to_do = ' & '.join(to_do)
            # Now we need to preorder
            if verbose:
                print('    Making a backup and %s file %i' % (to_do, ri + 1))
            raw = Raw(raw_file, preload=True, allow_maxshield=True)
            # rename split files if any
            regex = re.compile("-*.fif")
            split_files = glob.glob(raw_file[:-4] + regex.pattern)
            move_files = [raw_file] + split_files
            for f in move_files:
                move(f, f + '.orig')
            if need_reorder:
                raw._data[picks, :] = raw._data[picks, :][order]
            if need_anon:
                raw.info['subject_info'].update(anon)
            raw.info['description'] = write_key + anon_key
            raw.save(raw_file, format=raw.orig_format, overwrite=True)
        else:
            if verbose:
                print('    File %i already corrected' % (ri + 1))


def _all_files_fixed(p, subj, type_='pca'):
    """Determine if all files have been fixed for a subject"""
    return all(op.isfile(fname) and not any(_is_file_unfixed(fname)[:2])
               for fname in get_raw_fnames(p, subj, type_))


def _is_file_unfixed(fname, anon=None):
    """Determine if a file needs reordering or anonymization"""
    order = np.array([1, 2, 3, 5, 6, 7, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 30,
                      31, 32, 33, 34, 35, 36, 37, 38,
                      41, 42, 43, 44, 45, 46, 47, 48, 49,
                      51, 52, 54, 55, 56, 57, 58, 60,
                      39, 29, 18, 4, 8, 28, 40, 59, 50, 53]) - 1
    assert len(order) == 60
    write_key = 'LABSN_EEG_REORDER:' + ','.join([str(o) for o in order])
    anon_key = '' if anon is None else ';anonymized'
    with warnings.catch_warnings(record=True):
        raw = Raw(fname, preload=False, allow_maxshield=True)
    picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])
    if len(picks) == 0:
        return False, False, None, None, None, None
    if not len(picks) == len(order):
        raise RuntimeError('Incorrect number of EEG channels (%i) found '
                           'in %s' % (len(picks), op.basename(fname)))
    need_reorder = (write_key not in raw.info['description'])
    need_anon = (anon_key not in raw.info['description'])
    return need_reorder, need_anon, write_key, anon_key, picks, order
