# -*- coding: utf-8 -*-
# Copyright (c) 2014, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
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

The deidentified structural directories for the two subjects are needed
to do the forward and inverse solutions, extract them into your
SUBJECTS_DIR directory:

    * http://lester.ilabs.uw.edu/files/AKCLEE_107_slim.tar.gz
    * http://lester.ilabs.uw.edu/files/AKCLEE_110_slim.tar.gz

"""

import mnefun
from score import score
import numpy as np
from mne import set_log_level as log

log(verbose='WARNING')

params = mnefun.Params(tmin=-0.2, tmax=0.5, t_adjust=-4e-3,
                       n_jobs=18, n_jobs_mkl=1,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       decim=5, proj_sfreq=200, filter_length='5s')
params.subjects = ['subj_01', 'subj_02']
params.structurals = ['AKCLEE_107_slim', 'AKCLEE_110_slim']
params.dates = [(2014, 2, 14), (2014, 2, 10)]
params.score = score  # scoring function to use
params.subject_indices = np.arange(2)  # which subjects to run
params.plot_drop_logs = False  # turn off for demo or plots will block

params.acq_ssh = 'eric@172.28.161.8'  # minea
params.acq_dir = '/sinuhe/data01/eric_non_space'
params.sws_ssh = 'eric@172.25.148.15'  # kasga
params.sws_dir = '/data06/eric'

params.run_names = ['%s_funloc']
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.runs_empty = ['%s_erm']
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 2],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)

# The scoring function needs to produce an event file with these values
params.in_names = ['Aud', 'Vis', 'AudDev', 'VisDev']
params.in_numbers = [10, 20, 11, 21]

# These lines define how to translate the above event types into evoked files
params.analyses = [
    'All',
    'AV',
]
params.out_names = [
    ['All'],
    ['A', 'V'],
]
params.out_numbers = [
    [1, 1, 1, 1],    # Combine all trials
    [1, 2, -1, -1],  # Get auditory standards and visual standards
]
params.must_match = [
    [],
    [0, 1],  # we want the same number of auditory and visual trials
]

# Set what will run
mnefun.do_processing(
    params,
    fetch_raw=False,  # Fetch raw recording files from acq machine
    # Make SUBJ/raw_fif/SUBJ_prebad.txt file with space-separated
    # list of bad MEG channel numbers, needed for running SSS.
    push_raw=False,  # Push raw files and SSS script to SSS workstation
    do_sss=False,  # Run SSS remotely
    fetch_sss=False,  # Fetch SSSed files
    do_score=False,  # do scoring
    do_ch_fix=False,  # Fix channel ordering
    # Examine SSS'ed files and make SUBJ/bads/bad_ch_SUBJ_post-sss.txt,
    # usually only contains EEG channels, needed for preprocessing.
    gen_ssp=False,  # Generate SSP vectors
    apply_ssp=False,  # Apply SSP vectors and filtering
    gen_covs=False,  # Generate covariances
    # Make SUBJ/trans/SUBJ-trans.fif file in mne_analyze, needed for fwd calc.
    gen_fwd=False,  # Generate forward solutions (and source space if needed)
    gen_inv=False,  # Generate inverses
    write_epochs=False,  # Write epochs to disk
    gen_report=True,  # Write mne report html to disk
)
