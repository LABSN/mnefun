# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import warnings
import mne

from ._paths import get_raw_fnames, get_event_fnames


def default_score(p, subjects, run_indices):
    """Default scoring function that just passes event numbers through"""
    for si, subj in enumerate(subjects):
        print('  Scoring subject %s... ' % subj)

        # Figure out what our filenames should be
        raw_fnames = get_raw_fnames(p, subj, 'raw', False, False,
                                    run_indices[si])
        eve_fnames = get_event_fnames(p, subj, run_indices[si])

        for raw_fname, eve_fname in zip(raw_fnames, eve_fnames):
            with warnings.catch_warnings(record=True):
                raw = mne.io.read_raw_fif(raw_fname, allow_maxshield='yes')
            events = mne.find_events(raw, stim_channel='STI101',
                                     shortest_event=1)
            mne.write_events(eve_fname, events)
