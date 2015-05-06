# -*- coding: utf-8 -*-
# Copyright (c) 2014, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
This sample scoring script shows how to convert the serial binary stamping
from expyfun into meaningful event numbers using mnefun, and then write
out the data to the location mnefun expects.
"""

from __future__ import print_function

import os
import numpy as np
from os import path as op
import mne
from mnefun import extract_expyfun_events


# Original coding used 8XX8 to code event types, here we translate to
# a nicer decimal scheme
_expyfun_dict = {
    10: 10,  # 8448  (9) + 1 = 10: auditory std, recode as 10
    12: 11,  # 8488 (11) + 1 = 12: visual std, recode as 11
    14: 20,  # 8848 (13) + 1 = 14: auditory dev, recode as 20
    16: 21,  # 8888 (15) + 1 = 16: visual dev, recode as 21
}


def score(p, subjects):
    """Scoring function"""
    for subj in subjects:
        print('  Running subject %s... ' % subj, end='')

        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)

        for run_name in p.run_names:
            fname = op.join(p.work_dir, subj, p.raw_dir,
                            (run_name % subj) + p.raw_fif_tag)
            events, presses = extract_expyfun_events(fname)[:2]
            for ii in range(len(events)):
                events[ii, 2] = _expyfun_dict[events[ii, 2]]
            fname_out = op.join(out_dir,
                                'ALL_' + (run_name % subj) + '-eve.lst')
            mne.write_events(fname_out, events)

            # get subject performance
            devs = (events[:, 2] % 2 == 1)
            has_presses = np.array([len(pr) > 0 for pr in presses], bool)
            n_devs = np.sum(devs)
            hits = np.sum(has_presses[devs])
            fas = np.sum(has_presses[~devs])
            misses = n_devs - hits
            crs = (len(devs) - n_devs) - fas
            print('HMFC: %s, %s, %s, %s' % (hits, misses, fas, crs))
