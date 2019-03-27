# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from os import path as op
import glob
import numpy as np
import re
import warnings
from shutil import move

from mne.io import Raw, read_raw_fif
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
            if isinstance(raw_file, str):
                raw = read_raw_fif(raw_file, preload=True,
                                   allow_maxshield='yes')
            else:
                raw = raw_file
            # rename split files if any
            regex = re.compile("-*.fif")
            if isinstance(raw_file, str):
                split_files = glob.glob(raw_file[:-4] + regex.pattern)
                move_files = [raw_file] + split_files
                for f in move_files:
                    move(f, f + '.orig')
            if need_reorder:
                raw._data[picks, :] = raw._data[picks, :][order]
            if need_anon and raw.info['subject_info'] is not None:
                raw.info['subject_info'].update(anon)
            raw.info['description'] = write_key + anon_key
            if isinstance(raw_file, str):
                raw.save(raw_file, fmt=raw.orig_format, overwrite=True)
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
        if isinstance(fname, Raw):
            raw = fname
        else:
            raw = read_raw_fif(fname, preload=False, allow_maxshield='yes')
    picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])
    if len(picks) == 0:
        return False, False, None, None, None, None
    if not len(picks) == len(order):
        raise RuntimeError('Incorrect number of EEG channels (%i) found '
                           'in %s' % (len(picks), op.basename(fname)))
    need_reorder = (write_key not in raw.info['description'])
    need_anon = (anon_key not in raw.info['description'])
    return need_reorder, need_anon, write_key, anon_key, picks, order


ch_names_uw_70 = [
    'Fp1', 'Fpz', 'Fp2',
    'AF3', 'AFz', 'AF4',  # AF7/8
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',  # FT9/10
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',  # T9/10
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',  # TP9/10
    'P9', 'P7', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P8', 'P10',  # P5/6
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'Iz',
    ]
ch_names_uw_60 = [
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT9', 'FT7', 'FC5', 'FC1', 'FC2', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP3', 'CP1', 'CP2', 'CP4', 'TP8', 'TP10',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'Iz']
assert len(ch_names_uw_70) == len(ch_names_uw_60) == 60
