# -*- coding: utf-8 -*-
# Copyright (c) 2014, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
----------------------------------
Example experiment analysis script
----------------------------------

This sample script shows how to preprocess a simple MEG experiment
from start to finish.

The experiment was a simple audio/visual oddball detection task. One
potential purpose would be e.g. functional localization of auditory and
visual cortices.

Note that you will need to change the "acq_ssh" and "sss_ssh" parameters
to reflect your username/password on the relevant machines. You will also
need to set up public key authentication between your machine and the
two remote machines (acquisition/minea and SSS/kasga). Tutorial here:

    * https://help.ubuntu.com/community/SSH/OpenSSH/Keys

The deidentified structural directories for the one subject is needed
to do the forward and inverse solutions, extract this into your
SUBJECTS_DIR directory:

    * http://lester.ilabs.uw.edu/files/AKCLEE_110_slim.tar.gz

"""

import mnefun
from score import score
import numpy as np

try:
    # Use niprov as handler for events if it's installed
    from niprov.mnefunsupport import handler
except ImportError:
    handler = None

params = mnefun.Params(tmin=-0.2, tmax=0.5, t_adjust=-4e-3,
                       n_jobs=6, n_jobs_mkl=1, cov_method='shrunk',
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       decim=5, proj_sfreq=200, filter_length='auto')

params.subjects = ['subj_01', 'subj_02']
params.structurals = ['AKCLEE_107_slim', 'AKCLEE_110_slim']
params.dates = [(2014, 2, 14), None]  # Use "None" to more fully anonymize
params.score = score  # Scoring function used to slice data into trials
params.subject_indices = np.arange(2)  # Define which subjects to run
params.plot_drop_logs = False  # Turn off so plots do not halt processing
# typically this is a single string, but here we'll use a per-subject dict
# to demonstrate that both surface and volumetric source spaces can be used
params.src = dict(
    subj_01='oct6',  # standard oct-6 surface
    subj_02='vol5',  # 5 mm volumetric grid
)
# The default is to use the median (across runs) of the starting head positions
# individually for each subject.
# params.trans_to = 'median'

# You can also use a translation, plus x-axis rotation (-30 means backward 30°)
# params.trans_to = (0., 0., 0.05, -30)

# Or you can transform to the time-weighted average head pos
# for each subject individually.
params.trans_to = 'twa'

# Set the parameters for head position estimation:
params.coil_t_window = 'auto'  # use the smallest reasonable window size
params.coil_t_step_min = 0.01  # this is generally a good value
params.coil_dist_limit = 0.005  # same as MaxFilter, can be made more lenient

# Data can be annotated for omission (from epoching and destination head
# position calculation) by setting parameters like these (these are quite
# stringent!)
params.rotation_limit = 0.2  # deg/s
params.translation_limit = 0.0001  # m/s
# remove segments with < 3 good coils for at least 100 ms
params.coil_bad_count_duration_limit = 0.1

# Set parameters for remotely connecting to acquisition computer
params.acq_ssh = 'minea'  # Could also be e.g., "eric@minea.ilabs.uw.edu"
# Pass list of paths to search and fetch raw data
params.acq_dir = ['/sinuhe_data01/eric_non_space',
                  '/data101/eric_non_space',
                  '/sinuhe/data01/eric_non_space',
                  '/sinuhe/data02/eric_non_space',
                  '/sinuhe/data03/eric_non_space']

# Parameters for remotely connecting to SSS workstation ('sws') can be set
# by adding a file ~/.mnefun/mnefun.json with contents like:
#
#     $ mkdir ~/.mnefun
#     $ echo '{"sws_ssh":"kasga","sws_dir":"/data06/larsoner/sss_work","sws_port":22}' > ~/.mnefun/mnefun.json
#
# This should be preferred to the old way, which was to set in each script
# when running on your machine:
#
#     params.sws_ssh = 'kasga'
#     params.sws_dir = '/data06/larsoner/sss_work'
#
# Using per-machine config files rather than per-script variables should
# help increase portability of scripts without hurting reproducibility
# (assuming we all use the same version of MaxFilter, which should be a
# safe assumption).

# Set the niprov handler to deal with events:
params.on_process = handler

params.run_names = ['%s_funloc']
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.runs_empty = ['%s_erm']  # Define empty room runs
params.compute_rank = True  # compute rank of the noise covariance matrix
params.cov_rank = None  # preserve cov rank when using advanced estimators
params.force_erm_cov_rank_full = False  # compute and use the empty-room rank

# Define number of SSP projectors.
# Three lists, one for ECG/EOG/continuous, each list with entries for
# Grad/Mag/EEG. Can also be a per-subject dict (or defaultdict), like:
params.proj_nums = dict(
    subj_01=[[2, 2, 0], [1, 1, 3], [0, 0, 0]],
    subj_02=[[2, 2, 0], [1, 1, 2], [0, 0, 0]],
    )
params.proj_meg = 'combined'  # jointly estimate MEG projectors
params.proj_ave = True  # better projections by averaging ECG/EOG epochs
params.eog_f_lims = [1, 10]  # band-pass limits for the EOG detection+artifacts

# Set to True to use Autoreject module to compute noisy sensor thresholds for
# epoching
params.autoreject_thresholds = False
# Set to ('meg', 'eeg', eog') to reject trials based on EOG
params.autoreject_types = ('mag', 'grad', 'eeg')
# To define noisy sensor thresholds set to 'auto' to compute values with
# Autoreject module # or e.g., # dict(grad=1500e-13, mag=5000e-15, eeg=150e-6).
params.auto_bad_reject = None
params.cov_method = 'shrunk'  # Cleaner noise covariance regularization
# python | maxfilter for choosing SSS applied using either Maxfilter or MNE
params.sss_type = 'python'
# Run remote Maxfilter to automatically detect bad channels prior to SSS
params.mf_autobad = True
# Threshold limit for Maxfilter noisy channel detection.
params.mf_badlimit = 7
# The scoring function needs to produce an event file with these values
params.in_numbers = [10, 11, 20, 21]
# Those values correspond to real categories as:
params.in_names = ['Auditory/Standard', 'Visual/Standard',
                   'Auditory/Deviant', 'Visual/Deviant']

# Define how to translate the above event types into evoked files
params.analyses = [
    'All',
    'AV',
]
params.out_names = [
    ['All'],
    params.in_names,
]
params.out_numbers = [
    [1, 1, 1, 1],       # Combine all trials
    params.in_numbers,  # Leave events split the same way they were scored
]
params.must_match = [
    [],
    [0, 1],  # Only ensure the standard event counts match
]

params.report_params.update(  # add a couple of nice diagnostic plots
    good_hpi_count=True,  # Number of good HPI coils (default True)
    chpi_snr=True,  # cHPI SNR (default True)
    head_movement=True,  # head movement (default True)
    raw_segments=True,  # 10 evenly spaced raw data segments (default True)
    psd=False,  # often slow (default True)
    ssp_topomaps=True,  # SSP topomaps (default True)
    source_alignment=True,  # source alignment (defiault True)
    drop_log=True,  # plot the epochs drop log (default True)
    snr=dict(analysis='All', name='All',
             inv='%s-55-sss-meg-eeg-free-inv.fif'),
    bem=True,  # BEM layers
    covariance=True,  # covariance image and SVD plots
    whitening=dict(analysis='All', name='All',
                   cov='%s-55-sss-cov.fif'),
    sensor=dict(analysis='All', name='All', times=[0.1, 0.2]),
    source=dict(analysis='All', name='All',
                inv='%s-55-sss-meg-eeg-free-inv.fif', times=[0.09, 0.4],
                views='lat', size=(800, 400)),
)

# Set what processing steps will execute
default = False
mnefun.do_processing(
    params,
    fetch_raw=default,     # Fetch raw recording files from acquisition machine
    do_score=default,      # Do scoring to slice data into trials

    # Before running SSS, make SUBJ/raw_fif/SUBJ_prebad.txt file with
    # space-separated list of bad MEG channel numbers
    push_raw=default,      # Push raw files and SSS script to SSS workstation
    do_sss=default,        # Run SSS remotely (on sws) or locally with MNE
    fetch_sss=default,     # Fetch SSSed files from SSS workstation
    do_ch_fix=default,     # Fix channel ordering

    # Before running SSP, examine SSS'ed files and make
    # SUBJ/bads/bad_ch_SUBJ_post-sss.txt; usually, this should only contain EEG
    # channels.
    gen_ssp=default,       # Generate SSP vectors
    apply_ssp=default,     # Apply SSP vectors and filtering
    write_epochs=default,  # Write epochs to disk
    gen_covs=default,      # Generate covariances

    # Make SUBJ/trans/SUBJ-trans.fif using mne_analyze; needed for fwd calc.
    gen_fwd=default,       # Generate forward solutions (and source space)
    gen_inv=default,       # Generate inverses
    gen_report=default,    # Write mne report html of results to disk
    print_status=default,  # Print completeness status update
)
