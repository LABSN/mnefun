# -*- coding: utf-8 -*-
"""
-------------
Simulate data
-------------

This example shows how to use mnefun to simulate data with head movements.

Note: you need to run the ``analysis_fun.py`` example to have the
necessary raw data.
"""

# Todo:
#
#     * Add noise to this example
#     * Add line noise
#     * Add more realistic ECG/EOG (not magnetic dipoles)
#     * Add command-line interface

import os.path as op
import warnings
import numpy as np
import matplotlib.pyplot as plt

from mne import (get_config, read_source_spaces, SourceEstimate,
                 read_labels_from_annot, get_chpi_positions)
from mne.io import read_info, Raw, calculate_chpi_positions
from mnefun import simulate_movement

pulse_tmin, pulse_tmax = 0., 0.1

this_dir = op.dirname(__file__)
subjects_dir = get_config('SUBJECTS_DIR')
subj, subject = 'subj_01', 'AKCLEE_107_slim'

# This is a position file that has been modified/truncated for speed
fname_pos = op.join(this_dir, '%s_funloc_hp_trunc.txt' % subj)

# Simulate some data
data_dir = op.join(this_dir, 'funloc', subj)
bem_dir = op.join(subjects_dir, subject, 'bem')
fname_raw = op.join(data_dir, 'raw_fif', '%s_funloc_raw.fif' % subj)
fname_erm = op.join(data_dir, 'raw_fif', '%s_erm_raw.fif' % subj)
trans = op.join(data_dir, 'trans', '%s-trans.fif' % subj)
bem = op.join(bem_dir, '%s-5120-5120-5120-bem-sol.fif' % subject)
src = read_source_spaces(op.join(bem_dir, '%s-oct-6-src.fif' % subject))
sfreq = read_info(fname_raw, verbose=False)['sfreq']

# construct appropriate STC
print('Constructing original (simulated) sources')
tmin, tmax = -0.2, 0.8
vertices = [s['vertno'] for s in src]
n_vertices = sum(s['nuse'] for s in src)
data = np.ones((n_vertices, int((tmax - tmin) * sfreq)))
stc = SourceEstimate(data, vertices, -0.2, 1. / sfreq, subject)

# limit activation to a square pulse in time at two vertices in space
labels = [read_labels_from_annot(subject, 'aparc.a2009s', hemi,
                                 regexp='G_temp_sup-G_T_transv')[0]
          for hi, hemi in enumerate(('lh', 'rh'))]
stc = stc.in_label(labels[0] + labels[1])
stc.data.fill(0)
stc.data[:, np.where(np.logical_and(stc.times >= pulse_tmin,
                                    stc.times <= pulse_tmax))[0]] = 10e-9

# Simulate data with movement, with no noise (cov=None) for simplicity
with warnings.catch_warnings(record=True):
    raw = Raw(fname_raw, allow_maxshield=True, preload=True)
print('Simulating data')
raw_movement = simulate_movement(raw, fname_pos, stc, trans, src, bem,
                                 interp='zero', n_jobs=6)

# Simulate data with no movement (use initial head position)
raw_stationary = simulate_movement(raw, None, stc, trans, src, bem,
                                   interp='zero', n_jobs=6)

# Extract positions
out = calculate_chpi_positions(raw_movement, verbose=True)
t_move, trans_move = out[:, 0], out[:, 4:7]

out = calculate_chpi_positions(raw_stationary, verbose=True)
t_stat, trans_stat = out[:, 0], out[:, 4:7]

trans_orig, rot_orig, t_orig = get_chpi_positions(fname_pos)
t_orig -= raw.first_samp / raw.info['sfreq']

# Let's look at the results, just translation for simplicity
axes = 'XYZ'
plt.figure()
ts = [t_stat, t_orig, t_move]
transs = [trans_stat, trans_orig, trans_move]
labels = ['stationary', 'original', 'simulated']
sizes = [5, 10, 5]
colors = 'ykr'
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
