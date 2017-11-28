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
                       n_jobs=6, n_jobs_mkl=1,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       decim=5, proj_sfreq=200, filter_length='5s')

params.subjects = ['subj_01', 'subj_02']
params.structurals = [None, 'AKCLEE_110_slim']  # None means use sphere
params.dates = [(2014, 2, 14), None]  # Use "None" to more fully anonymize
params.score = score  # Scoring function used to slice data into trials
params.subject_indices = np.arange(2)  # Define which subjects to run
params.plot_drop_logs = False  # Turn off so plots do not halt processing

# Set parameters for remotely connecting to acquisition computer
params.acq_ssh = 'minea'  # Could also be e.g., "eric@minea.ilabs.uw.edu"
# Pass list of paths to search and fetch raw data
params.acq_dir = ['/sinuhe_data01/eric_non_space',
                  '/data101/eric_non_space',
                  '/sinuhe/data01/eric_non_space',
                  '/sinuhe/data02/eric_non_space',
                  '/sinuhe/data03/eric_non_space']

# Set parameters for remotely connecting to SSS workstation ('sws')
params.sws_ssh = 'kasga'
params.sws_dir = '/data06/larsoner'

# Set the niprov handler to deal with events:
params.on_process = handler

params.run_names = ['%s_funloc']
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.runs_empty = ['%s_erm']  # Define empty room runs

# Define number of SSP projectors. Columns correspond to Grad/Mag/EEG chans
params.proj_nums = [[1, 1, 0],  # ECG
                    [1, 1, 2],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)
# By default SSP projection scalp topography maps will be saved in
# sss_pca_folder for inspection. To avoid having images saved to disk set
# params.plot_pca = False
params.autoreject_thresholds = True  # Set to True to use Autoreject module to set global epoch rejection thresholds  # noqa
params.cov_method = 'shrunk'  # Cleaner noise covariance regularization
# python | maxfilter for choosing SSS applied using either Maxfilter or MNE
params.sss_type = 'python'
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

# Set what processing steps will execute
mnefun.do_processing(
    params,
    fetch_raw=False,     # Fetch raw recording files from acquisition machine
    do_score=False,      # Do scoring to slice data into trials

    # Before running SSS, make SUBJ/raw_fif/SUBJ_prebad.txt file with
    # space-separated list of bad MEG channel numbers
    push_raw=False,      # Push raw files and SSS script to SSS workstation
    do_sss=False,        # Run SSS remotely (on sws) or locally with mne-python
    fetch_sss=False,     # Fetch SSSed files from SSS workstation
    do_ch_fix=False,     # Fix channel ordering

    # Before running SSP, examine SSS'ed files and make
    # SUBJ/bads/bad_ch_SUBJ_post-sss.txt; usually, this should only contain EEG
    # channels.
    gen_ssp=False,       # Generate SSP vectors
    apply_ssp=False,     # Apply SSP vectors and filtering
    plot_psd=False,      # Plot raw data power spectra
    write_epochs=False,  # Write epochs to disk
    gen_covs=False,      # Generate covariances

    # Make SUBJ/trans/SUBJ-trans.fif using mne_analyze; needed for fwd calc.
    gen_fwd=False,       # Generate forward solutions (and src space if needed)
    gen_inv=False,       # Generate inverses
    gen_report=True,    # Write mne report html of results to disk
    print_status=True,  # Print completeness status update
)
