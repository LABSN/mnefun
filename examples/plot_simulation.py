# -*- coding: utf-8 -*-
"""
-------------
Simulate data
-------------

This example shows how to use mnefun to simulate data with head movements.

Note: you need to run the ``analysis_fun.py`` example to have the
necessary raw data.
"""

import os.path as op
import warnings
import numpy as np

from mne import (get_config, read_source_spaces, SourceEstimate,
                 read_labels_from_annot, find_events, Epochs,
                 compute_proj_raw, compute_raw_data_covariance)
from mne.io import read_info, Raw
from mne.viz import plot_topo
from mnefun import simulate_movement

snr = 25.
pulse_tmin, pulse_tmax = 0., 0.1

this_dir = op.dirname(__file__)
subjects_dir = get_config('SUBJECTS_DIR')
subj, subject = 'subj_01', 'AKCLEE_107_slim'

# This is a position file that has been modified/truncated for speed
fname_pos = op.join(this_dir, '%s_funloc_hp_trunc.txt' % subj)

data_dir = op.join(this_dir, 'funloc', subj)
bem_dir = op.join(subjects_dir, subject, 'bem')
fname_raw = op.join(data_dir, 'raw_fif', '%s_funloc_raw.fif' % subj)
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
                                    stc.times <= pulse_tmax))[0]] = 1e-8

# Let's used a slightly cleaned up noise covariance
print('Cleaning data a little bit')
with warnings.catch_warnings(record=True):
    raw = Raw(fname_raw, allow_maxshield=True, preload=True)
raw.add_proj(compute_proj_raw(raw, n_mag=2, n_grad=2, duration=None))
cov = compute_raw_data_covariance(raw)

# Simulate data with movement
print('Simulating data')
raw_movement = simulate_movement(
    raw, fname_pos, stc, trans, src, bem, cov=cov,
    snr=snr, snr_tmin=pulse_tmin, snr_tmax=pulse_tmax,
    n_jobs=6)

# Simulate data with no movement (use original head position)
raw_stationary = simulate_movement(
    raw, None, stc, trans, src, bem, cov=cov,
    snr=snr, snr_tmin=pulse_tmin, snr_tmax=pulse_tmax,
    n_jobs=6)

# Let's look at the data
print('Plotting evoked data')
evokeds = list()
for raw, title in zip((raw_movement, raw_stationary),
                      ('Moving', 'Stationary')):
    events = find_events(raw, 'STI101')
    epochs = Epochs(raw, events, dict(title=1), tmin, tmax, verbose=False)
    evokeds.append(epochs.average())
plot_topo(evokeds, color=['r', 'w'])
