.. _overview:

========
Overview
========

The processing pipeline steps and relationships are given below.
All YAML parameters are described in their appropriate sections.

.. contents:: Contents
   :depth: 3

Flow chart
----------

.. graphviz:: _static/flow.dot

.. note::
     At the very least, the following are missing from this flowchart:

     - ``-annot.fif`` creation
     - custom ``-annot.fif``
     - maxbad bad-channel computation and output
     - autoreject support
     - references to :mod:`mne` functions used
     - extra projectors

Preparing your machine for head position estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parameters for remotely connecting to SSS workstation ('sws') can be set
by adding a file ~/.mnefun/mnefun.json with contents like:

.. code-block:: console

    $ mkdir ~/.mnefun
    $ echo '{"sws_ssh":"kasga", "sws_dir":"/data06/larsoner/sss_work", "sws_port":22}' > ~/.mnefun/mnefun.json

This should be preferred to the old way, which was to set in each script
when running on your machine::

    params.sws_ssh = 'kasga'
    params.sws_dir = '/data06/larsoner/sss_work'

Using per-machine config files rather than per-script variables should
help increase portability of scripts without hurting reproducibility
(assuming we all use the same version of MaxFilter, which should be a
safe assumption).

Running parameters
------------------

``general``: General options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
work_dir : str
    Working directory, usually ".".
subjects_dir : str
    Directory containing the structurals.
subject_indices : list
    Which subjects to process.
disp_files : bool
    Display status.


1. fetch_raw
------------

Fetch raw files from an acquisition machine.

``fetch_raw``: Raw fetching parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
subjects : list of str
    Subject names.
structurals : list of str
    List of subject structurals.
dates : list of tuple or None
    Dates to use for anonymization. Use "None" to more fully anonymize.
acq_ssh : str
     The acquisition machine SSH name.
acq_dir : list of str
    List of paths to search and fetch raw data.
acq_port : int
     Acquisition port.
run_names : list of str
    Run names for the paradigm.
runs_empty : list of str
    Empty room run names.
subject_run_indices : list of array-like | None
    Run indices to include for each subject. This can be a list
    (must be same length as ``params.subjects``) or a dict (keys are subject
    strings, values are the run indices) including a defaultdict. None is an
    alias for "all runs".

2. do_score
-----------

Do the scoring. This converts TTL triggers to meaningful events.

``scoring``: Scoring parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
score : callable | None
     Scoring function used to slice data into trials.
on_process : callable
    Called at each processing step.

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

``preprocessing: multithreading``: Multithreading parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_jobs : int
    Number of jobs to use in parallel operations.
n_jobs_mkl : int
    Number of jobs to spawn in parallel for operations that can make
    use of MKL threading. If Numpy/Scipy has been compiled with MKL
    support, it is best to leave this at 1 or 2 since MKL will
    automatically spawn threads. Otherwise, n_cpu is a good choice.
n_jobs_fir : int | str
    Number of threads to use for FIR filtering. Can also be 'cuda'
    if the system supports CUDA.
n_jobs_resample : int | str
    Number of threads to use for resampling. Can also be 'cuda'
    if the system supports CUDA.

``preprocessing: bads``: Bad-channel parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mf_autobad : bool
    Default False. If True use Maxfilter to automatically mark bad
    channels prior to SSS denoising.
mf_badlimit : int
    MaxFilter threshold for noisy channel detection (default is 7).
auto_bad : float | None
    If not None, bad channels will be automatically excluded if
    they disqualify a proportion of events exceeding ``autobad``.
auto_bad_reject : str | dict | None
    Default is None. Must be defined if using Autoreject module to
    compute noisy sensor rejection criteria. Set to 'auto' to compute
    criteria automatically, or dictionary of channel keys and amplitude
    values e.g., dict(grad=1500e-13, mag=5000e-15, eeg=150e-6) to define
    rejection threshold(s). See
    http://autoreject.github.io/ for details.
auto_bad_flat : dict | None
    Flat threshold for auto bad.
auto_bad_eeg_thresh : float | None
    Threshold for auto_bad EEG detection.
auto_bad_meg_thresh : float | None
    Threshold for auto_bad MEG detection.

``preprocessing: head_position_estimation``: Head position estimation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
coil_t_window : float
    Time window for coil position estimation.
coil_t_step_min : float
    Coil step min for head / cHPI coil position estimation.
coil_dist_limit : float
    Dist limit for coils.

``preprocessing: annotations``: Annotation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
coil_bad_count_duration_limit : float
    Remove segments with < 3 good coils for at least this many sec.
rotation_limit : float
    Rotation limit (deg/s) for annotating bad segments.
translation_limit : float
    Head translation limit (m/s) for annotating bad segments.

``preprocessing: sss``: SSS parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
movecomp : str | None
    Movement compensation to use. Can be 'inter' or None.
sss_type : str
    signal space separation method. Must be either 'maxfilter' or 'python'
int_order : int
    Order of internal component of spherical expansion. Default is 8.
    Value of 6 recomended for infant data.
ext_order : int
    Order of external component of spherical expansion. Default is 3.
sss_regularize : str
    SSS regularization, usually "in".
tsss_dur : float | None
    Buffer length (in seconds) fpr Spatiotemporal SSS. Default is 60.
    however based on system specification a shorter buffer may be
    appropriate. For data containing excessive head movements e.g. young
    children a buffer size of 4s is recommended.
st_correlation : float
    Correlation limit between inner and outer subspaces used to reject
    ovwrlapping intersecting inner/outer signals during spatiotemporal SSS.
    Default is .98 however a smaller value of .9 is recommended for infant/
    child data.
filter_chpi : str
    Filter cHPI signals before SSS.
trans_to : str | array-like, (3,) | None
    The destination location for the head. Can be:

    'median' (default)
        Median (across runs) of the starting head positions.
    'twa'
        Time-weighted average head position.
    ``None``
        Will not change the head position.
    str
        Path to a FIF file containing a MEG device to head transformation.
    array-like
        First three elements are coordinates to translate to.
        An optional fourth element gives the x-axis rotation (e.g., -30 means
        a backward 30° rotation).
sss_origin : array-like, shape (3,) | str
    Origin of internal and external multipolar moment space in meters.
    Default is center of sphere fit to digitized head points.
dig_with_eeg : bool
    If True, include EEG points in estimating the head origin.
ct_file : str
    Cross-talk file, usually "uw" to auto-load the UW file.
cal_file : str
    Calibration file, usually "uw" to auto-load the UW file.
sss_format : str
    Deprecated. SSS numerical format when using MaxFilter.
mf_args : str
    Deprecated. Extra arguments for MF SSS.


4. do_ch_fix
------------

Fix EEG channel ordering, and also anonymize files.

5. gen_ssp
----------

.. warning:: Before running SSP, examine SSS'ed files and make
             SUBJ/bads/bad_ch_SUBJ_post-sss.txt; usually, this should only
             contain EEG channels.

Generate SSP vectors.

``preprocessing: filtering``: Filtering parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hp_cut : float | None
    Highpass cutoff in Hz. Use None for no highpassing.
hp_trans : float
    High-pass transition band.
lp_cut : float
    Cutoff for lowpass filtering.
lp_trans : float
    Low-pass transition band.
filter_length : int | str
    See :func:`mne.filter.create_filter`.
fir_design : str
    See :func:`mne.filter.create_filter`.
fir_window : str
    See :func:`mne.filter.create_filter`.
phase : str
    See :func:`mne.filter.create_filter`.

``preprocessing: ssp``: SSP creation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
proj_nums : list | dict
    List of projector counts to use for ECG/EOG/ERM; each list contains
    three values for grad/mag/eeg channels.
    Can be a dict that maps subject names to projector counts to use.
proj_sfreq : float | None
    The sample freq to use for calculating projectors. Useful since
    time points are not independent following low-pass. Also saves
    computation to downsample.
proj_meg : str
    Can be "separate" (default for backward compat) or "combined"
    (should be better for SSS'ed data).
drop_thresh : float
    The percentage threshold to use when deciding whether or not to
    plot Epochs drop_log.
plot_raw : bool
    If True, plot the raw files with the ECG/EOG events overlaid.
ssp_eog_reject : dict | None
    Amplitude rejection criteria for EOG SSP computation. None will
    use the mne-python default.
ssp_ecg_reject : dict | None
    Amplitude rejection criteria for ECG SSP computation. None will
    use the mne-python default.
eog_channel : str | dict | None
    The channel to use to detect EOG events. None will use EOG* channels.
    In lieu of an EOG recording, MEG1411 may work.
ecg_channel : str | dict | None
    The channel to use to detect ECG events. None will use ECG063.
    In lieu of an ECG recording, MEG1531 may work.
    Can be a dict that maps subject names to channels.
eog_t_lims : tuple
    The time limits for EOG calculation. Default (-0.25, 0.25).
ecg_t_lims : tuple
    The time limits for ECG calculation. Default(-0.08, 0.08).
eog_f_lims : tuple
    Band-pass limits for EOG detection and calculation. Default (0, 2).
ecg_f_lims : tuple
    Band-pass limits for ECG detection and calculation. Default (5, 35).
eog_thresh : float | dict | None
    Threshold for EOG detection. Can vary per subject.
proj_ave : bool
    If True, average artifact epochs before computing proj.
proj_extra : str | None
    Extra projector filename to load for each subject.
get_projs_from : list of int
    Indices for runs to get projects from.
cont_lp : float
    Lowpass to use for continuous ERM projectors.
plot_drop_logs : bool
    If True, plot drop logs after preprocessing.


6. apply_ssp
------------
Apply SSP vectors and filtering to the files.


7. write_epochs
---------------
Write epochs to disk.

``epoching``: Epoching parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tmin : float
    tmin for events.
tmax : float
    tmax for events.
t_adjust : float
    Adjustment for delays (e.g., -4e-3 compensates for a 4 ms delay
    in the trigger.
baseline : tuple | None | str
    Baseline to use. If "individual", use ``params.bmin`` and
    ``params.bmax``, otherwise pass as the baseline parameter to
    mne-python Epochs. ``params.bmin`` and ``params.bmax`` will always
    be used for covariance calculation. This is useful e.g. when using
    a high-pass filter and no baselining is desired (but evoked
    covariances should still be calculated from the baseline period).
bmin : float
    Lower limit for baseline compensation.
bmax : float
    Upper limit for baseline compensation.
decim : int
    Amount to decimate the data after filtering when epoching data
    (e.g., a factor of 5 on 1000 Hz data yields 200 Hz data).
epochs_type : str | list
    Can be 'fif', 'mat', or a list containing both.
match_fun : callable | None
    If None, standard matching will be performed. If a function,
    must_match will be ignored, and ``match_fun`` will be called
    to equalize event counts.
reject : dict
    Rejection parameters for epochs.
flat : dict
    Flat thresholds for epoch rejection.
reject_tmin : float | None
    Reject minimum time to use when epoching. None will use ``tmin``.
reject_tmax : float | None
    Reject maximum time to use when epoching. None will use ``tmax``.
on_missing : string
    Can set to ‘error’ | ‘warning’ | ‘ignore’. Default is 'error'.
    Determine what to do if one or several event ids are not found in the
    recording during epoching. See mne.Epochs docstring for further
    details.
autoreject_thresholds : bool | False
    If True use autoreject module to compute global rejection thresholds
    for epoching. Make sure autoreject module is installed. See
    http://autoreject.github.io/ for instructions.
autoreject_types : tuple
    Default is ('mag', 'grad', 'eeg'). Can set to ('mag', 'grad', 'eeg',
    'eog) to use EOG channel rejection criterion from autoreject module to
    reject trials on basis of EOG.
reject_epochs_by_annot : bool
    If True, reject epochs by BAD annotations.
analyses : list of str
    Lists of analyses of interest.
in_names : list of str
    Names of input events.
in_numbers : list of list of int
    Event numbers (in scored event files) associated with each name.
out_names : list of list of str
    Event types to make out of old ones.
out_numbers : list of list of int
    Event numbers to convert to (e.g., [[1, 1, 2, 3, 3], ...] would create
    three event types, where the first two and last two event types from
    the original list get collapsed over).
must_match : list of int
    Indices from the original in_names that must match in event counts
    before collapsing. Should eventually be expanded to allow for
    ratio-based collapsing.

8. gen_covs
-----------
Generate covariances.

``covariance``: Covariance parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cov_method : str
    Covariance calculation method.
compute_rank : bool
    Default is False. Set to True to compute rank of the noise covariance
    matrix during inverse kernel computation.
pick_events_cov : callable | None
    Function for picking covariance events.
cov_rank : str | int
    Cov rank to use, usually "auto".
cov_rank_tol : float
    Tolerance for covariance rank computation.
force_erm_cov_rank_full : bool
    If True, force the ERM cov to be full rank.
    Usually not needed, but might help when the empty-room data
    is short and/or there are a lot of head movements.


9. gen_fwd
----------
.. warning:: Make SUBJ/trans/SUBJ-trans.fif using :ref:`mne:gen_mne_coreg`.

Generate forward solutions (and source space if necessary).

``forward``: Forward parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bem_type : str
    Defaults to ``'5120-5120-5120'``, use ``'5120'`` for a
    single-layer BEM.
src : str | dict
    Can start be:

    - 'oct6' to use a surface source space decimated using the 6th
      (or another integer) subdivision of an octahedron, or
    - 'vol5' to use a volumetric grid source space with 5mm (or another
      integer) spacing
src_pos : float
    Default is 7 mm. Defines source grid spacing for volumetric source
    space.
fwd_mindist : float
    Minimum distance (mm) for sources in the brain from the skull in order
    for them to be included in the forward solution source space.

10. gen_inv
-----------

Generate inverses.

``inverse``: Inverse parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
inv_names : list of str
    Inverse names to use.
inv_runs : list of int
    Runs to use for each inverse.


11. gen_report
--------------

Write :class:`mne.Report` HTML of results to disk.

``report_params``: Report parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
chpi_snr : bool
    cHPI SNR (default True).
good_hpi_count : bool
    Number of good HPI coils (default True).
head_movement : bool
    Head movement (default True).
raw_segments : bool
    10 evenly spaced raw data segments (default True).
psd : bool
    Raw PSDs, often slow (default True).
ssp_topomaps : bool
    SSP topomaps (default True).
source_alignment : bool
    Source alignment (default True).
drop_log : bool
    Plot the epochs drop log (default True).
bem : bool
    Covariance image and SVD plots.
snr : dict
    SNR plots, with keys 'analysis', 'name', and 'inv'.
whitening : dict
    Whitening plots, with keys 'analysis', 'name', and 'cov'.
sensor : dict
    Sensor topomaps, with keys 'analysis', 'name', and 'times'.
source : dict
    Source plots, with keys 'analysis', 'name', 'inv', 'times', 'views',
    and 'size'.


Filename standardization
------------------------
mnefun imposes custom standardized structure on filenames:

``naming``: File naming tags and folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
list_dir : str
    Directory for event lists, usually "lists".
bad_dir : str
    Directory to use for bad channels, usually "bads".
bad_tag : str
    Tag for bid channel filename, usually "_post-sss.txt".
raw_dir : str
    Raw directory, usually "raw_fif".
keep_orig : bool
    Keep original files after anonymization.
raw_fif_tag : str
    File tag for raw data, usually "_raw.fif".
sss_fif_tag : str
    File tag for SSS-processed files, usually "_raw_sss.fif".
sss_dir : str
    Directory to use for SSS processed files, usually "sss_fif".
pca_dir : str
    Directory for processed files, usually "sss_pca_fif".
epochs_dir : str
    Directory for epochs, usually "epochs".
epochs_prefix : str
    The prefix to use for the ``-epo.fif`` file.
epochs_tag : str
    Tag for epoochs, usually '-epo'.
eq_tag : str
    Tag for equalized data, usually "eq".
cov_dir : str
    Directory to use for covariances, usually "covariance".
forward_dir : str
    Directory for forward solutions, usually "forward".
trans_dir : str
    Directory to use for trans files, usually "trans".
inverse_dir : str
    Directory for storing inverses, usually "inverse".
inv_tag : str
    Tag for all inverses, usually "-sss".
inv_erm_tag : str
    Tag for ERM inverse, usually "-erm".
inv_fixed_tag : str
    Tag for fixed inverse, usually "-fixed".
inv_loose_tag : str
    Tag for loose inverse, usually "".
inv_free_tag : str
    Tag for free orientation inverse, usually "-free".
