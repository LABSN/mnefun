# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import os
import warnings
from os import path as op
import mne


def default_score(p, subjects):
    """Default scoring function that just passes event numbers through"""
    for subj in subjects:
        print('  Scoring subject %s... ' % subj)

        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)

        for run_name in p.run_names:
            fname = op.join(p.work_dir, subj, p.raw_dir,
                            (run_name % subj) + p.raw_fif_tag)
            with warnings.catch_warnings(record=True):
                raw = mne.io.read_raw_fif(fname, allow_maxshield=True)
            events = mne.find_events(raw, stim_channel='STI101',
                                     shortest_event=1)
            fname_out = op.join(out_dir,
                                'ALL_' + (run_name % subj) + '-eve.lst')
            mne.write_events(fname_out, events)
