========
Overview
========

The processing pipeline steps and relationships are given below.

.. note::
     At the very least, the following are missing:

     - ``-annot.fif`` creation
     - custom ``-annot.fif``
     - maxbad bad-channel computation and output
     - autoreject support
     - references to :mod:`mne` functions used
     - extra projectors
     - YAML schema options/list

.. contents:: Contents
   :depth: 2

Flow chart
----------

.. graphviz:: _static/flow.dot

1. fetch_raw
------------

Fetches raw files from an acquisition machine.

2. do_score
-----------

Do the scoring. This converts TTL triggers to meaningful events.

.. _do_sss:

3. do_sss
---------

.. warning:: Before running SSS, make SUBJ/raw_fif/SUBJ_prebad.txt file with
             space-separated list of bad MEG channel numbers.

Run SSS processing. This will:

1. Copy each file to the SSS workstation to estimate head positions.
2. Copy the head positions to the local machine.
3. Delete the file from the remote machine.
4. Run SSS processing locally using :func:`mne.preprocessing.maxwell_filter`.

4. do_ch_fix
------------

Fix EEG channel ordering, and also anonymize files.

5. gen_ssp
----------

.. warning:: Before running SSP, examine SSS'ed files and make
             SUBJ/bads/bad_ch_SUBJ_post-sss.txt; usually, this should only
             contain EEG channels.

Generate SSP vectors.

6. apply_ssp
------------
Apply SSP vectors and filtering to the files.

7. write_epochs
---------------
Write epochs to disk.

8. gen_covs
-----------
Generate covariances.

9. gen_fwd
----------
.. warning:: Make SUBJ/trans/SUBJ-trans.fif using :ref:`mne:gen_mne_coreg`.

Generate forward solutions (and source space if necessary).

10. gen_inv
-----------

Generate inverses.

11. gen_report
--------------

Write :class:`mne.Report` HTML of results to disk.
