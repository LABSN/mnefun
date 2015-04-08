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
import niprov.mnefunsupport

params = mnefun.Params(tmin=-0.2, tmax=0.5, t_adjust=-4e-3,
                       n_jobs=6, n_jobs_mkl=1,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       decim=5, proj_sfreq=200, filter_length='5s')
params.subjects = ['subj_01', 'subj_02']
params.structurals = ['AKCLEE_107_slim', 'AKCLEE_110_slim']
params.dates = [(2014, 2, 14), (2014, 2, 10)]
params.score = score  # scoring function to use
params.subject_indices = np.arange(2)  # which subjects to run
params.plot_drop_logs = False  # turn off for demo or plots will block

params.acq_ssh = 'minea'  # can also be e.g., "eric@minea.ilabs.uw.edu"
params.acq_dir = '/sinuhe/data01/eric_non_space'
params.sws_ssh = 'kasga'
params.sws_dir = '/data06/larsoner'

## set the niprov handler to deal with events:
params.on_process = niprov.mnefunsupport.handler 

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
    fetch_raw=True,  # Fetch raw recording files from acq machine
    do_score=True,  # do scoring
    # Make SUBJ/raw_fif/SUBJ_prebad.txt file with space-separated
    # list of bad MEG channel numbers, needed for running SSS.
    push_raw=True,  # Push raw files and SSS script to SSS workstation
    do_sss=True,  # Run SSS remotely
    fetch_sss=True,  # Fetch SSSed files
    do_ch_fix=True,  # Fix channel ordering
    # Examine SSS'ed files and make SUBJ/bads/bad_ch_SUBJ_post-sss.txt,
    # usually only contains EEG channels, needed for preprocessing.
    gen_ssp=True,  # Generate SSP vectors
    apply_ssp=True,  # Apply SSP vectors and filtering
    write_epochs=True,  # Write epochs to disk
    gen_covs=True,  # Generate covariances
    # Make SUBJ/trans/SUBJ-trans.fif file in mne_analyze, needed for fwd calc.
    gen_fwd=True,  # Generate forward solutions (and source space if needed)
    gen_inv=True,  # Generate inverses
    gen_report=True,  # Write mne report html of results to disk
    print_status=True,  # Print completeness status update
)
