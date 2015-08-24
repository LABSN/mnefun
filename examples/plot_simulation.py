# -*- coding: utf-8 -*-
"""
-------------
Simulate data
-------------

This example shows how to use mnefun to simulate data with head movements.

Note: you need to run the ``analysis_fun.py`` example to have the
necessary raw data.

The resulting positions were re-estimated using e.g.::

    $ maxfilter -f move_raw.fif -headpos -hp hp_move.txt

"""

import os.path as op
import warnings
import numpy as np
import matplotlib.pyplot as plt

from mne import (get_config, read_source_spaces, SourceEstimate,
                 read_labels_from_annot, get_chpi_positions)
from mne.io import read_info, Raw
from mnefun import simulate_movement

pulse_tmin, pulse_tmax = 0., 0.1

this_dir = op.dirname(__file__)
subjects_dir = get_config('SUBJECTS_DIR')
subj, subject = 'subj_01', 'AKCLEE_107_slim'

# This is a position file that has been modified/truncated for speed
fname_pos_orig = op.join(this_dir, '%s_funloc_hp_trunc.txt' % subj)
# These were generated using Maxfilter after simulation
fname_pos_move = op.join(this_dir, 'hp_move.txt')
fname_pos_stat = op.join(this_dir, 'hp_stat.txt')

# Set up paths
data_dir = op.join(this_dir, 'funloc', subj)
bem_dir = op.join(subjects_dir, subject, 'bem')
fname_raw = op.join(data_dir, 'raw_fif', '%s_funloc_raw.fif' % subj)
trans = op.join(data_dir, 'trans', '%s-trans.fif' % subj)
bem = op.join(bem_dir, '%s-5120-5120-5120-bem-sol.fif' % subject)
src = read_source_spaces(op.join(bem_dir, '%s-oct-6-src.fif' % subject))
sfreq = read_info(fname_raw, verbose=False)['sfreq']

# ############################################################################
# construct appropriate brain activity

print('Constructing original (simulated) sources')
tmin, tmax = -0.2, 0.8
vertices = [s['vertno'] for s in src]
n_vertices = sum(s['nuse'] for s in src)
data = np.zeros((n_vertices, int((tmax - tmin) * sfreq)))
stc = SourceEstimate(data, vertices, -0.2, 1. / sfreq, subject)

# limit activation to a square pulse in time at two vertices in space
labels = [read_labels_from_annot(subject, 'aparc.a2009s', hemi,
                                 regexp='G_temp_sup-G_T_transv')[0]
          for hi, hemi in enumerate(('lh', 'rh'))]
stc = stc.in_label(labels[0] + labels[1])
stc.data.fill(0)
stc.data[:, (stc.times >= pulse_tmin) & (stc.times <= pulse_tmax)] = 10e-9

# ############################################################################
# Simulate data

# Simulate data with movement
with warnings.catch_warnings(record=True):
    raw = Raw(fname_raw, allow_maxshield=True)
raw_movement = simulate_movement(raw, fname_pos_orig, stc, trans, src, bem,
                                 interp='zero', n_jobs=6, verbose=True)

# Simulate data with no movement (use initial head position)
raw_stationary = simulate_movement(raw, None, stc, trans, src, bem,
                                   interp='zero', n_jobs=6, verbose=True)

# Extract positions
trans_move, rot_move, t_move = get_chpi_positions(fname_pos_move)
trans_stat, rot_stat, t_stat = get_chpi_positions(fname_pos_stat)
trans_orig, rot_orig, t_orig = get_chpi_positions(fname_pos_orig)

# ############################################################################
# Let's look at the results, just translation for simplicity

axes = 'XYZ'
fig = plt.figure(dpi=200)
ts = [t_orig, t_stat, t_move]
transs = [trans_orig, trans_stat, trans_move]
labels = ['original', 'stationary', 'simulated']
sizes = [10, 5, 5]
colors = 'kyr'
for ai, axis in enumerate(axes):
    ax = plt.subplot(3, 1, ai + 1)
    lines = []
    for t, data, size, color, label in zip(ts, transs, sizes, colors, labels):
        lines.append(ax.step(t, 1000 * data[:, ai], color=color, marker='o',
                             markersize=size, where='post', label=label)[0])
    ax.set_ylabel(axis)
    if ai == 1:
        plt.legend(lines)
    if ai == 2:
        ax.set_xlabel('Time (sec)')
    ax.set_axes('tight')

raw_movement.plot(lowpass=40., clipping='clamp')
