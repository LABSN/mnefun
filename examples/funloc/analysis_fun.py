# -*- coding: utf-8 -*-
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

Setting up a config file
------------------------
Parameters for remotely connecting to SSS workstation ('sws') can be set
by adding a file ~/.mnefun/mnefun.json with contents like:

    $ mkdir ~/.mnefun
    $ echo '{"sws_ssh":"kasga","sws_dir":"/data06/larsoner/sss_work","sws_port":22}' > ~/.mnefun/mnefun.json

This should be preferred to the old way, which was to set in each script
when running on your machine:

    params.sws_ssh = 'kasga'
    params.sws_dir = '/data06/larsoner/sss_work'

Using per-machine config files rather than per-script variables should
help increase portability of scripts without hurting reproducibility
(assuming we all use the same version of MaxFilter, which should be a
safe assumption).
"""  # noqa: E501

import mnefun
from score import score

params = mnefun.read_params('funloc_params.yml')
params.score = score
params.subject_indices = [0, 1]

# Set what processing steps will execute
default = False
mnefun.do_processing(
    params,
    fetch_raw=default,     # Fetch raw recording files from acquisition machine
    do_score=default,      # Do scoring to slice data into trials

    # Before running SSS, make SUBJ/raw_fif/SUBJ_prebad.txt file with
    # space-separated list of bad MEG channel numbers
    do_sss=default,        # Run SSS locally with MNE
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
