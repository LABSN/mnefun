# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import print_function

import os
import os.path as op
from contextlib import contextmanager
from copy import deepcopy
import warnings
from shutil import move, copy2
import subprocess
from collections import Counter
import time
import sys

import numpy as np
from scipy import linalg, io as spio
from numpy.testing import assert_allclose

import mne
from mne import (
    compute_proj_raw, make_fixed_length_events, Epochs, find_events,
    read_events, write_events, concatenate_events, read_cov,
    compute_covariance, write_cov, read_forward_solution,
    convert_forward_solution, write_proj, read_proj, setup_source_space,
    make_forward_solution, write_evokeds, make_sphere_model,
    setup_volume_source_space, pick_info, write_source_spaces,
    read_source_spaces, write_forward_solution, DipoleFixed,
    read_annotations)
from mne.externals.h5io import read_hdf5, write_hdf5

try:
    from mne import compute_raw_covariance  # up-to-date mne-python
except ImportError:  # oldmne-python
    from mne import compute_raw_data_covariance as compute_raw_covariance
from mne.preprocessing.ssp import compute_proj_ecg, compute_proj_eog
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.preprocessing.maxwell import (maxwell_filter,
                                       _trans_sss_basis,
                                       _get_mf_picks, _prep_mf_coils,
                                       _check_regularize,
                                       _regularize)
from mne.utils import verbose, logger

try:
    # Experimental version
    from mne.preprocessing.maxwell import _prep_regularize
except ImportError:
    _prep_regularize = None
from mne.bem import _check_origin
from mne.minimum_norm import make_inverse_operator
from mne.label import read_label
from mne.epochs import combine_event_ids
from mne.chpi import (filter_chpi, read_head_pos, write_head_pos,
                      _get_hpi_info, _get_hpi_initial_fit, _setup_hpi_struct,
                      _fit_cHPI_amplitudes, _fit_magnetic_dipole)
from mne.io.proj import _needs_eeg_average_ref_proj

from mne.cov import regularize
try:
    from mne.chpi import quat_to_rot, rot_to_quat
except ImportError:
    try:
        from mne.chpi import (_quat_to_rot as quat_to_rot,
                              _rot_to_quat as rot_to_quat)
    except ImportError:
        from mne.io.chpi import (_quat_to_rot as quat_to_rot,
                                 _rot_to_quat as rot_to_quat)
from mne.io import read_raw_fif, concatenate_raws, read_info, write_info
from mne.io.base import _annotations_starts_stops
from mne.io.constants import FIFF
from mne.io.pick import pick_types_forward, pick_types
from mne.io.meas_info import _empty_info
from mne.minimum_norm import write_inverse_operator
from mne.utils import run_subprocess, _time_mask
from mne.viz import plot_drop_log, tight_layout
from mne.fixes import _get_args as get_args
from mne.externals.six import string_types

from ._paths import (get_raw_fnames, get_event_fnames,
                     get_epochs_evokeds_fnames, safe_inserter, _regex_convert)
from ._status import print_proc_status
from ._reorder import fix_eeg_channels
from ._report import gen_html_report
from ._scoring import default_score

try:
    from functools import reduce
except Exception:
    pass


# Class adapted from:
# http://stackoverflow.com/questions/3603502/

class Frozen(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise AttributeError('%r is not an attribute of class %s. Call '
                                 '"unfreeze()" to allow addition of new '
                                 'attributes' % (key, self))
        object.__setattr__(self, key, value)

    def freeze(self):
        """Freeze the object so that only existing properties can be set"""
        self.__isfrozen = True

    def unfreeze(self):
        """Unfreeze the object so that additional properties can be added"""
        self.__isfrozen = False


# noinspection PyUnresolvedReferences
class Params(Frozen):
    """Make a parameter structure for use with `do_processing`

    This is technically a class, but it doesn't currently have any methods
    other than init.

    Parameters
    ----------
    tmin : float
        tmin for events.
    tmax : float
        tmax for events.
    t_adjust : float
        Adjustment for delays (e.g., -4e-3 compensates for a 4 ms delay
        in the trigger.
    bmin : float
        Lower limit for baseline compensation.
    bmax : float
        Upper limit for baseline compensation.
    n_jobs : int
        Number of jobs to use in parallel operations.
    lp_cut : float
        Cutoff for lowpass filtering.
    decim : int
        Amount to decimate the data after filtering when epoching data
        (e.g., a factor of 5 on 1000 Hz data yields 200 Hz data).
    proj_sfreq : float | None
        The sample freq to use for calculating projectors. Useful since
        time points are not independent following low-pass. Also saves
        computation to downsample.
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
    filter_length : int
        Filter length to use in FIR filtering. Longer filters generally
        have better roll-off characteristics, but more ringing.
    drop_thresh : float
        The percentage threshold to use when deciding whether or not to
        plot Epochs drop_log.
    epochs_type : str | list
        Can be 'fif', 'mat', or a list containing both.
    fwd_mindist : float
        Minimum distance for sources in the brain from the skull in order
        for them to be included in the forward solution source space.
    bem_type : str
        Defaults to ``'5120-5120-5120'``, use ``'5120'`` for a
        single-layer BEM.
    auto_bad : float | None
        If not None, bad channels will be automatically excluded if
        they disqualify a proportion of events exceeding ``autobad``.
    ecg_channel : str | None
        The channel to use to detect ECG events. None will use ECG063.
        In lieu of an ECG recording, MEG1531 may work.
    eog_channel : str
        The channel to use to detect EOG events. None will use EOG*.
        In lieu of an EOG recording, MEG1411 may work.
    plot_raw : bool
        If True, plot the raw files with the ECG/EOG events overlaid.
    match_fun : function | None
        If None, standard matching will be performed. If a function,
        must_match will be ignored, and ``match_fun`` will be called
        to equalize event counts.
    hp_cut : float | None
        Highpass cutoff in Hz. Use None for no highpassing.
    cov_method : str
        Covariance calculation method.
    ssp_eog_reject : dict | None
        Amplitude rejection criteria for EOG SSP computation. None will
        use the mne-python default.
    ssp_ecg_reject : dict | None
        Amplitude rejection criteria for ECG SSP computation. None will
        use the mne-python default.
    baseline : tuple | None | str
        Baseline to use. If "individual", use ``params.bmin`` and
        ``params.bmax``, otherwise pass as the baseline parameter to
        mne-python Epochs. ``params.bmin`` and ``params.bmax`` will always
        be used for covariance calculation. This is useful e.g. when using
        a high-pass filter and no baselining is desired (but evoked
        covariances should still be calculated from the baseline period).
    reject_tmin : float | None
        Reject minimum time to use when epoching. None will use ``tmin``.
    reject_tmax : float | None
        Reject maximum time to use when epoching. None will use ``tmax``.
    lp_trans : float
        Low-pass transition band.
    hp_trans : float
        High-pass transition band.

    Attributes
    ----------
    movecomp : str | None
        Movement compensation to use. Can be 'inter' or None.
    sss_type : str
        signal space separation method. Must be either 'maxfilter' or 'python'
    int_order : int
        Order of internal component of spherical expansion. Default is 8.
    ext_order : int
        Order of external component of spherical expansion. Default is 3.
        Value of 6 recomended for infant data
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
    trans_to : str | array-like, (3,) | None
        The destination location for the head. Can be ``None``, which
        will not change the head position, a string path to a FIF file
        containing a MEG device to head transformation, or a 3-element
        array giving the coordinates to translate to (with no rotations).
        Default is median head position across runs.
    sss_origin : array-like, shape (3,) | str
        Origin of internal and external multipolar moment space in meters.
        Default is center of sphere fit to digitized head points.
    fir_design : str
        Can be "firwin2" or "firwin".
    autoreject_thresholds : bool | False
        If True use autoreject module to compute global rejection thresholds
        for epoching. Make sure autoreject module is installed. See
        http://autoreject.github.io/ for instructions.
    autoreject_types : tuple
        Default is ('mag', 'grad', 'eeg'). Can set to ('mag', 'grad', 'eeg',
        'eog) to use EOG channel rejection criterion from autoreject module to
        reject trials on basis of EOG.
    src_pos : float
        Default is 7 mm. Defines source grid spacing for volumetric source
        space.
    on_missing : string
        Can set to ‘error’ | ‘warning’ | ‘ignore’. Default is 'error'.
        Determine what to do if one or several event ids are not found in the
        recording during epoching. See mne.Epochs docstring for further
        details.
    compute_rank : bool
        Default is False. Set to True to compute rank of the noise covariance
        matrix during inverse kernel computation.

    Returns
    -------
    params : instance of Params
        The parameters to use.

    See also
    --------
    do_processing
    mne.preprocessing.maxwell_filter

    Notes
    -----
    Params has additional properties. Use ``dir(params)`` to see
    all the possible options.
    """

    def __init__(self, tmin=None, tmax=None, t_adjust=0, bmin=-0.2, bmax=0.0,
                 n_jobs=6, lp_cut=55, decim=5, proj_sfreq=None, n_jobs_mkl=1,
                 n_jobs_fir='cuda', n_jobs_resample='cuda',
                 filter_length=32768, drop_thresh=1,
                 epochs_type='fif', fwd_mindist=2.0,
                 bem_type='5120-5120-5120', auto_bad=None,
                 ecg_channel=None, eog_channel=None,
                 plot_raw=False, match_fun=None, hp_cut=None,
                 cov_method='empirical', ssp_eog_reject=None,
                 ssp_ecg_reject=None, baseline='individual',
                 reject_tmin=None, reject_tmax=None,
                 lp_trans=0.5, hp_trans=0.5):
        self.reject = dict(eog=np.inf, grad=1500e-13, mag=5000e-15, eeg=150e-6)
        self.flat = dict(eog=0, grad=1e-13, mag=1e-15, eeg=1e-6)
        if ssp_eog_reject is None:
            ssp_eog_reject = dict(grad=2000e-13, mag=3000e-15,
                                  eeg=500e-6, eog=np.inf)
        if ssp_ecg_reject is None:
            ssp_ecg_reject = dict(grad=2000e-13, mag=3000e-15,
                                  eeg=50e-6, eog=250e-6)
        self.ssp_eog_reject = ssp_eog_reject
        self.ssp_ecg_reject = ssp_ecg_reject
        self.tmin = tmin
        self.tmax = tmax
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax
        self.t_adjust = t_adjust
        self.baseline = baseline
        self.bmin = bmin
        self.bmax = bmax
        self.run_names = None
        self.inv_names = None
        self.inv_runs = None
        self.work_dir = os.getcwd()
        self.n_jobs = n_jobs
        self.n_jobs_mkl = n_jobs_mkl
        self.n_jobs_fir = n_jobs_fir  # Jobs when using method='fir'
        self.n_jobs_resample = n_jobs_resample
        self.filter_length = filter_length
        self.cont_lp = 5
        self.lp_cut = lp_cut
        self.hp_cut = hp_cut
        self.lp_trans = lp_trans
        self.hp_trans = hp_trans
        self.phase = 'zero-double'
        self.fir_window = 'hann'
        self.fir_design = 'firwin2'
        self.disp_files = True
        self.plot_drop_logs = False  # plot drop logs after do_preprocessing_
        self.proj_sfreq = proj_sfreq
        self.decim = decim
        self.drop_thresh = drop_thresh
        self.bem_type = bem_type
        self.match_fun = match_fun
        if isinstance(epochs_type, string_types):
            epochs_type = (epochs_type,)
        if not all([t in ('mat', 'fif') for t in epochs_type]):
            raise ValueError('All entries in "epochs_type" must be "mat" '
                             'or "fif"')
        self.epochs_type = epochs_type
        self.fwd_mindist = fwd_mindist
        self.auto_bad = auto_bad
        self.auto_bad_reject = None
        self.auto_bad_flat = None
        self.auto_bad_meg_thresh = 10
        self.auto_bad_eeg_thresh = 10
        self.ecg_channel = ecg_channel
        self.eog_channel = eog_channel
        self.plot_raw = plot_raw

        # add standard file tags

        self.epochs_dir = 'epochs'
        self.cov_dir = 'covariance'
        self.inverse_dir = 'inverse'
        self.forward_dir = 'forward'
        self.list_dir = 'lists'
        self.trans_dir = 'trans'
        self.bad_dir = 'bads'
        self.raw_dir = 'raw_fif'
        self.sss_dir = 'sss_fif'
        self.pca_dir = 'sss_pca_fif'

        self.epochs_tag = '-epo'
        self.inv_tag = '-sss'
        self.inv_fixed_tag = '-fixed'
        self.inv_loose_tag = ''
        self.inv_free_tag = '-free'
        self.inv_erm_tag = '-erm'
        self.eq_tag = 'eq'
        self.sss_fif_tag = '_raw_sss.fif'
        self.bad_tag = '_post-sss.txt'
        self.keep_orig = False
        # This is used by fix_eeg_channels to fix original files
        self.raw_fif_tag = '_raw.fif'
        self.cal_file = None
        self.ct_file = None
        # SSS denoising params
        self.sss_type = 'maxfilter'
        self.mf_args = ''
        self.tsss_dur = 60.
        self.trans_to = 'median'  # where to transform head positions to
        self.sss_format = 'float'  # output type for MaxFilter
        self.movecomp = 'inter'
        self.int_order = 8
        self.ext_order = 3
        self.st_correlation = .98
        self.sss_origin = 'auto'
        self.sss_regularize = 'in'
        self.filter_chpi = True
        # boolean for whether data set(s) have an individual mri
        self.on_process = None
        # Use more than EXTRA points to fit headshape
        self.dig_with_eeg = False
        # Function to pick a subset of events to use to make a covariance
        self.pick_events_cov = None
        self.cov_method = cov_method
        self.proj_extra = None
        # These should be overridden by the user unless they are only doing
        # a small subset, e.g. epoching
        self.subjects = []
        self.structurals = None
        self.dates = None
        self.score = None  # defaults to passing events through
        self.acq_ssh = self.acq_dir = None
        self.acq_port = 22
        self.sws_ssh = self.sws_dir = None
        self.sws_port = 22
        self.subject_indices = []
        self.get_projs_from = []
        self.runs_empty = []
        self.proj_nums = [[0] * 3] * 3
        self.in_names = []
        self.in_numbers = []
        self.analyses = []
        self.out_names = []
        self.out_numbers = []
        self.must_match = []
        self.on_missing = 'error'  # for epochs
        self.subject_run_indices = None
        self.autoreject_thresholds = False
        self.autoreject_types = ('mag', 'grad', 'eeg')
        self.subjects_dir = None
        self.src_pos = 7.
        self.report_params = dict(
            good_hpi_count=True,
            head_movement=True,
            psd=True,
            ssp_topomaps=True,
            source_alignment=True,
            bem=True,
            source=None,
            )
        self.rotation_limit = np.inf
        self.translation_limit = np.inf
        self.coil_bad_count_duration_limit = np.inf  # for annotations
        self.coil_dist_limit = 0.005
        self.coil_t_window = 0.2  # default is same as MF
        self.coil_t_step_min = 0.01
        self.proj_ave = False
        self.compute_rank = False
        self.freeze()

    @property
    def pca_extra(self):
        return '_allclean_fil%d' % self.lp_cut

    @property
    def pca_fif_tag(self):
        return self.pca_extra + self.sss_fif_tag

    def convert_subjects(self, subj_template, struc_template=None):
        """Helper to convert subject names

        Parameters
        ----------
        subj_template : str
            Subject template to use.
        struc_template : str
            Structural template to use.
        """
        if struc_template is not None:
            if isinstance(struc_template, string_types):
                def fun(x):
                    return struc_template % x
            else:
                fun = struc_template
            new = [fun(subj) for subj in self.subjects]
            assert all(isinstance(subj, string_types) for subj in new)
            self.structurals = new
        if isinstance(subj_template, string_types):
            def fun(x):
                return subj_template % x
        else:
            fun = subj_template
        new = [fun(subj) for subj in self.subjects]
        assert all(isinstance(subj, string_types) for subj in new)
        self.subjects = new


def _get_baseline(p):
    """Helper to extract baseline from params"""
    if p.baseline == 'individual':
        baseline = (p.bmin, p.bmax)
    else:
        baseline = p.baseline
    return baseline


def do_processing(p, fetch_raw=False, do_score=False, push_raw=False,
                  do_sss=False, fetch_sss=False, do_ch_fix=False,
                  gen_ssp=False, apply_ssp=False,
                  write_epochs=False, gen_covs=False, gen_fwd=False,
                  gen_inv=False, gen_report=False, print_status=True):
    """Do M/EEG data processing

    Parameters
    ----------
    p : instance of Params
        The parameter structure.
    fetch_raw : bool
        Fetch raw recording files from acquisition machine.
    do_score : bool
        Do scoring.
    push_raw : bool
        Push raw recording files to SSS workstation.
    do_sss : bool
        Run SSS remotely on SSS workstation.
    fetch_sss : bool
        Fetch SSS files from SSS workstation.
    do_ch_fix : bool
        Fix channel ordering.
    gen_ssp : bool
        Generate SSP vectors.
    apply_ssp : bool
        Apply SSP vectors and filtering.
    write_epochs : bool
        Write epochs to disk.
    gen_covs : bool
        Generate covariances.
    gen_fwd : bool
        Generate forward solutions.
    get_inv : bool
        Generate inverses.
    gen_report : bool
        Generate HTML reports.
    print_status : bool
        Print status (determined from file structure).
    """
    # Generate requested things
    if p.sss_type == 'python':
        push_raw = False
        fetch_sss = False
    bools = [fetch_raw,
             do_score,
             push_raw,
             do_sss,
             fetch_sss,
             do_ch_fix,
             gen_ssp,
             apply_ssp,
             write_epochs,
             gen_covs,
             gen_fwd,
             gen_inv,
             gen_report,
             print_status,
             ]
    texts = ['Pulling raw files from acquisition machine',
             'Scoring subjects',
             'Pushing raw files to remote workstation',
             'Running SSS using %s' % p.sss_type,
             'Pulling SSS files from remote workstation',
             'Fixing EEG order',
             'Preprocessing files',
             'Applying preprocessing',
             'Doing epoch EQ/DQ',
             'Generating covariances',
             'Generating forward models',
             'Generating inverse solutions',
             'Generating HTML Reports',
             'Status',
             ]
    score_fun = p.score if p.score is not None else default_score
    if len(get_args(score_fun)) == 2:
        score_fun_two = score_fun

        def score_fun(p, subjects, run_indices):
            return score_fun_two(p, subjects)
    funcs = [fetch_raw_files,
             score_fun,
             push_raw_files,
             run_sss,
             fetch_sss_files,
             fix_eeg_files,
             do_preprocessing_combined,
             apply_preprocessing_combined,
             save_epochs,
             gen_covariances,
             gen_forwards,
             gen_inverses,
             gen_html_report,
             print_proc_status,
             ]
    assert len(bools) == len(texts) == len(funcs)

    # Only run a subset of subjects

    n_subj_orig = len(p.subjects)

    sinds = p.subject_indices
    if sinds is None:
        sinds = np.arange(len(p.subjects))

    subjects = np.array(p.subjects)[sinds].tolist()

    structurals = p.structurals
    if structurals is not None:
        assert len(structurals) == n_subj_orig
        structurals = np.array(structurals)[sinds].tolist()

    dates = p.dates
    if dates is not None:
        assert len(dates) == n_subj_orig
        dates = [tuple([int(dd) for dd in d]) if d is not None else None
                 for d in np.array(p.dates)[sinds]]

    decim = p.decim
    if not isinstance(decim, (list, tuple)):
        decim = [decim] * len(p.subjects)
    assert len(decim) == n_subj_orig
    decim = np.array(decim)
    assert decim.dtype.char in 'il', decim.dtype
    assert decim.shape == (len(p.subjects),), decim.shape
    decim = decim[sinds]

    run_indices = p.subject_run_indices
    if run_indices is None:
        run_indices = [None] * len(p.subjects)
    assert len(run_indices) == len(p.subjects)
    run_indices = [r for ri, r in enumerate(run_indices) if ri in sinds]
    assert all(r is None or np.in1d(r, np.arange(len(p.run_names))).all()
               for r in run_indices)

    # Actually do the work

    outs = [None] * len(bools)
    for ii, (b, text, func) in enumerate(zip(bools, texts, funcs)):
        if b:
            t0 = time.time()
            print(text + '. ')
            if func is None:
                raise ValueError('function is None')
            if func == fix_eeg_files:
                outs[ii] = func(p, subjects, structurals, dates, run_indices)
            elif func in (gen_forwards, gen_html_report):
                outs[ii] = func(p, subjects, structurals, run_indices)
            elif func == save_epochs:
                outs[ii] = func(p, subjects, p.in_names, p.in_numbers,
                                p.analyses, p.out_names, p.out_numbers,
                                p.must_match, decim, run_indices)
            elif func == print_proc_status:
                outs[ii] = func(p, subjects, structurals, p.analyses,
                                run_indices)
            else:
                outs[ii] = func(p, subjects, run_indices)
            print('  (' + timestring(time.time() - t0) + ')')
            if p.on_process is not None:
                p.on_process(text, func, outs[ii], p)
    print("Done")


def _is_dir(d):
    """Safely check for a directory (allowing symlinks)"""
    return op.isdir(op.abspath(d))


def fetch_raw_files(p, subjects, run_indices):
    """Fetch remote raw recording files (only designed for *nix platforms)"""
    for si, subj in enumerate(subjects):
        print('  Checking for proper remote filenames for %s...' % subj)
        subj_dir = op.join(p.work_dir, subj)
        if not _is_dir(subj_dir):
            os.mkdir(subj_dir)
        raw_dir = op.join(subj_dir, p.raw_dir)
        if not op.isdir(raw_dir):
            os.mkdir(raw_dir)
        fnames = get_raw_fnames(p, subj, 'raw', True, False,
                                run_indices[si])
        assert len(fnames) > 0
        # build remote raw file finder
        if isinstance(p.acq_dir, string_types):
            use_dir = [p.acq_dir]
        else:
            use_dir = p.acq_dir
        finder_stem = 'find ' + ' '.join(use_dir)
        finder = (finder_stem + ' -o '.join([' -type f -regex ' +
                                             _regex_convert(f)
                                             for f in fnames]))
        # Ignore "Permission denied" errors:
        # https://unix.stackexchange.com/questions/42841/how-to-skip-permission-denied-errors-when-running-find-in-linux  # noqa
        finder += '2>&1 | grep -v "Permission denied"'
        stdout_ = run_subprocess(
            ['ssh', '-p', str(p.acq_port), p.acq_ssh, finder],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)[0]
        remote_fnames = [x.strip() for x in stdout_.splitlines()]
        if not any(fname.startswith(rd.rstrip('/') + '/') for rd in use_dir
                   for fname in remote_fnames):
            raise IOError('Unable to find files at remote locations. '
                          'Check filenames, for example:\n%s'
                          % remote_fnames[:1])
        # make the name "local" to the acq dir, so that the name works
        # remotely during rsync and locally during copyfile
        remote_dir = [fn[:fn.index(op.basename(fn))]
                      for fn in remote_fnames][0]
        remote_fnames = [op.basename(fname) for fname in remote_fnames]
        want = set(op.basename(fname) for fname in fnames)
        got = set(op.basename(fname) for fname in remote_fnames)
        if want != got.intersection(want):
            raise RuntimeError('Could not find all files, missing:\n' +
                               '\n'.join(sorted(want - got)))
        if len(remote_fnames) != len(fnames):
            warnings.warn('Found more files than expected on remote server.\n'
                          'Likely split files were found. Please confirm '
                          'results.')
        print('  Pulling %s files for %s...' % (len(remote_fnames), subj))
        cmd = ['rsync', '-ave', 'ssh -p %s' % p.acq_port,
               '--prune-empty-dirs', '--partial',
               '--include', '*/']
        for fname in remote_fnames:
            cmd += ['--include', op.basename(fname)]
        remote_loc = '%s:%s' % (p.acq_ssh, op.join(remote_dir, ''))
        cmd += ['--exclude', '*', remote_loc, op.join(raw_dir, '')]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # move files to root raw_dir
        for fname in remote_fnames:
            from_ = fname.index(subj)
            move(op.join(raw_dir, fname[from_:].lstrip('/')),
                 op.join(raw_dir, op.basename(fname)))
        # prune the extra directories we made
        for fname in remote_fnames:
            from_ = fname.index(subj)
            next_ = op.split(fname[from_:].lstrip('/'))[0]
            while len(next_) > 0:
                if op.isdir(op.join(raw_dir, next_)):
                    os.rmdir(op.join(raw_dir, next_))  # safe; goes if empty
                next_ = op.split(next_)[0]


def calc_median_hp(p, subj, out_file, ridx):
    """Calculate median head position"""
    print('        Estimating median head position ...')
    raw_files = get_raw_fnames(p, subj, 'raw', False, False, ridx)
    ts = []
    qs = []
    info = None
    for fname in raw_files:
        info = read_info(fname)
        trans = info['dev_head_t']['trans']
        ts.append(trans[:3, 3])
        m = trans[:3, :3]
        # make sure we are a rotation matrix
        assert_allclose(np.dot(m, m.T), np.eye(3), atol=1e-5)
        assert_allclose(np.linalg.det(m), 1., atol=1e-5)
        qs.append(rot_to_quat(m))
    assert info is not None
    if len(raw_files) == 1:  # only one head position
        dev_head_t = info['dev_head_t']
    else:
        t = np.median(np.array(ts), axis=0)
        rot = np.median(quat_to_rot(np.array(qs)), axis=0)
        trans = np.r_[np.c_[rot, t[:, np.newaxis]],
                      np.array([0, 0, 0, 1], t.dtype)[np.newaxis, :]]
        dev_head_t = {'to': 4, 'from': 1, 'trans': trans}
    info = _empty_info(info['sfreq'])
    info['dev_head_t'] = dev_head_t
    write_info(out_file, info)


def calc_twa_hp(p, subj, out_file, ridx):
    """Calculate time-weighted average head position."""
    if not p.movecomp:
        # Eventually we could relax this but probably YAGNI
        raise RuntimeError('Cannot use time-weighted average head position '
                           'when movecomp is off.')
    print('        Estimating time-weighted average head position ...')
    raw_fnames = get_raw_fnames(p, subj, 'raw', False, False, ridx)
    assert len(raw_fnames) >= 1
    norm = 0
    A = np.zeros((4, 4))
    pos = np.zeros(3)
    for raw_fname in raw_fnames:
        raw = mne.io.read_raw_fif(raw_fname, allow_maxshield='yes',
                                  verbose='error')
        hp, annot, _ = _head_pos_annot(p, raw_fname, prefix='          ')
        try:
            raw.set_annotations(annot)
        except AttributeError:
            raw.annotations = annot
        good = np.ones(len(raw.times))
        ts = np.concatenate((hp[:, 0],
                             [(raw.last_samp + 1) / raw.info['sfreq']]))
        ts -= raw.first_samp / raw.info['sfreq']
        idx = raw.time_as_index(ts, use_rounding=True)
        assert idx[-1] == len(good)
        # Mark times bad that are bad according to annotations
        onsets, ends = _annotations_starts_stops(raw, 'bad')
        for onset, end in zip(onsets, ends):
            good[onset:end] = 0
        dt = np.diff(np.cumsum(np.concatenate([[0], good]))[idx])
        dt = dt / raw.info['sfreq']
        del good, idx, ts
        pos += np.dot(dt, hp[:, 4:7])
        these_qs = hp[:, 1:4]
        res = 1 - np.sum(these_qs * these_qs, axis=-1, keepdims=True)
        assert (res >= 0).all()
        these_qs = np.concatenate((these_qs, np.sqrt(res)), axis=-1)
        assert np.allclose(np.linalg.norm(these_qs, axis=1), 1)
        these_qs *= dt[:, np.newaxis]
        # rank 1 update method
        # https://arc.aiaa.org/doi/abs/10.2514/1.28949?journalCode=jgcd
        # https://github.com/tolgabirdal/averaging_quaternions/blob/master/wavg_quaternion_markley.m  # noqa: E501
        # qs.append(these_qs)
        outers = np.einsum('ij,ik->ijk', these_qs, these_qs)
        A += outers.sum(axis=0)
        norm += dt.sum()
    A /= norm
    best_q = linalg.eigh(A)[1][:, -1]  # largest eigenvector is the wavg
    # Same as the largest eigenvector from the concatenation of all
    # best_q = linalg.svd(np.concatenate(qs).T)[0][:, 0]
    best_q = best_q[:3] * np.sign(best_q[-1])
    trans = np.eye(4)
    trans[:3, :3] = quat_to_rot(best_q)
    trans[:3, 3] = pos / norm
    dev_head_t = mne.Transform('meg', 'head', trans)
    info = _empty_info(raw.info['sfreq'])
    info['dev_head_t'] = dev_head_t
    write_info(out_file, info)


def push_raw_files(p, subjects, run_indices):
    """Push raw files to SSS workstation"""
    if len(subjects) == 0:
        return
    print('  Pushing raw files to SSS workstation...')
    # do all copies at once to avoid multiple logins
    copy2(op.join(op.dirname(__file__), 'run_sss.sh'), p.work_dir)
    includes = ['--include', op.sep + 'run_sss.sh']
    if not isinstance(p.trans_to, string_types):
        raise TypeError(' Illegal head transformation argument to MaxFilter.')
    elif p.trans_to not in ('default', 'median'):
        _check_trans_file(p)
        includes += ['--include', op.sep + p.trans_to]
    for si, subj in enumerate(subjects):
        subj_dir = op.join(p.work_dir, subj)
        raw_dir = op.join(subj_dir, p.raw_dir)

        out_pos = op.join(raw_dir, subj + '_center.txt')
        if not op.isfile(out_pos):
            print('    Determining head center for %s... ' % subj, end='')
            in_fif = op.join(raw_dir,
                             safe_inserter(p.run_names[0], subj) +
                             p.raw_fif_tag)
            if p.dig_with_eeg:
                dig_kinds = (FIFF.FIFFV_POINT_EXTRA, FIFF.FIFFV_POINT_LPA,
                             FIFF.FIFFV_POINT_NASION, FIFF.FIFFV_POINT_RPA,
                             FIFF.FIFFV_POINT_EEG)
            else:
                dig_kinds = (FIFF.FIFFV_POINT_EXTRA,)
            origin_head = fit_sphere_to_headshape(read_info(in_fif),
                                                  dig_kinds=dig_kinds,
                                                  units='mm')[1]
            out_string = ' '.join(['%0.0f' % np.round(number)
                                   for number in origin_head])
            with open(out_pos, 'w') as fid:
                fid.write(out_string)

        med_pos = op.join(raw_dir, subj + '_median_pos.fif')
        if not op.isfile(med_pos):
            calc_median_hp(p, subj, med_pos, run_indices[si])
        root = op.sep + subj
        raw_root = op.join(root, p.raw_dir)
        includes += ['--include', root, '--include', raw_root,
                     '--include', op.join(raw_root, op.basename(out_pos)),
                     '--include', op.join(raw_root, op.basename(med_pos))]
        prebad_file = _prebad(p, subj)
        includes += ['--include',
                     op.join(raw_root, op.basename(prebad_file))]
        fnames = get_raw_fnames(p, subj, 'raw', True, True, run_indices[si])
        assert len(fnames) > 0
        for fname in fnames:
            assert op.isfile(fname), fname
            includes += ['--include', op.join(raw_root, op.basename(fname))]
    assert ' ' not in p.sws_dir
    assert ' ' not in p.sws_ssh
    cmd = (['rsync', '-aLve', 'ssh -p %s' % p.sws_port, '--partial'] +
           includes + ['--exclude', '*'])
    cmd += ['.', '%s:%s' % (p.sws_ssh, op.join(p.sws_dir, ''))]
    run_subprocess(cmd, cwd=p.work_dir,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _check_trans_file(p):
    """Helper to make sure our trans_to file exists"""
    if not isinstance(p.trans_to, string_types):
        raise ValueError('trans_to must be a string')
    if p.trans_to not in ('default', 'median'):
        if not op.isfile(op.join(p.work_dir, p.trans_to)):
            raise ValueError('Trans position file "%s" not found'
                             % p.trans_to)


def run_sss(p, subjects, run_indices):
    """Run SSS preprocessing remotely (only designed for *nix platforms) or
    locally using Maxwell filtering in mne-python"""
    if p.sss_type == 'python':
        print('  Applying SSS locally using mne-python')
        run_sss_locally(p, subjects, run_indices)
    else:
        for si, subj in enumerate(subjects):
            files = get_raw_fnames(p, subj, 'raw', False, True,
                                   run_indices[si])
            n_files = len(files)
            files = ':'.join([op.basename(f) for f in files])
            erm = get_raw_fnames(p, subj, 'raw', 'only', True, run_indices[si])
            n_files += len(erm)
            erm = ':'.join([op.basename(f) for f in erm])
            erm = ' --erm ' + erm if len(erm) > 0 else ''
            assert isinstance(p.tsss_dur, float) and p.tsss_dur > 0
            st = ' --st %s' % p.tsss_dur
            if p.sss_format not in ('short', 'long', 'float'):
                raise RuntimeError('format must be short, long, or float')
            fmt = ' --format ' + p.sss_format
            assert p.movecomp in ['inter', None]
            mc = ' --mc %s' % str(p.movecomp).lower()
            _check_trans_file(p)
            trans = ' --trans ' + p.trans_to
            run_sss = (op.join(p.sws_dir, 'run_sss.sh') + st + fmt + trans +
                       ' --subject ' + subj + ' --files ' + files + erm + mc +
                       ' --args=\"%s\"' % p.mf_args)
            cmd = ['ssh', '-p', str(p.sws_port), p.sws_ssh, run_sss]
            s = 'Remote output for %s on %s files:' % (subj, n_files)
            print('-' * len(s))
            print(s)
            print('-' * len(s))
            run_subprocess(cmd, stdout=sys.stdout, stderr=sys.stderr)
            print('-' * 70, end='\n\n')


def fetch_sss_files(p, subjects, run_indices):
    """Pull SSS files (only designed for *nix platforms)"""
    if len(subjects) == 0:
        return
    includes = []
    for subj in subjects:
        includes += ['--include', subj,
                     '--include', op.join(subj, 'sss_fif'),
                     '--include', op.join(subj, 'sss_fif', '*'),
                     '--include', op.join(subj, 'sss_log'),
                     '--include', op.join(subj, 'sss_log', '*')]
    assert ' ' not in p.sws_dir
    assert ' ' not in p.sws_ssh
    cmd = (['rsync', '-ave', 'ssh -p %s' % p.sws_port, '--partial', '-K'] +
           includes + ['--exclude', '*'])
    cmd += ['%s:%s' % (p.sws_ssh, op.join(p.sws_dir, '*')), '.']
    run_subprocess(cmd, cwd=p.work_dir, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)


def run_sss_command(fname_in, options, fname_out, host='kasga', port=22,
                    fname_pos=None, stdout=None, stderr=None, prefix='',
                    work_dir='~/'):
    """Run Maxfilter remotely and fetch resulting file

    Parameters
    ----------
    fname_in : str
        The filename to process.
    options : str
        The command-line options for Maxfilter.
    fname_out : str | None
        Output filename to use to store the result on the local machine.
        None will output to a temporary file.
    host : str
        The SSH/scp host to run the command on.
    fname_pos : str | None
        The ``-hp fname_pos`` to use with MaxFilter.
    stdout : file-like | None
        Where to send stdout.
    stderr : file-like | None
        Where to send stderr.
    prefix : str
        The text to prefix to messages.
    work_dir : str
        Where to store the temporary files.
    """
    # let's make sure we can actually write where we want
    if not op.isfile(fname_in):
        raise IOError('input file not found: %s' % fname_in)
    if not op.isdir(op.dirname(op.abspath(fname_out))):
        raise IOError('output directory for output file does not exist')
    if any(x in options for x in ('-f ', '-o ', '-hp ')):
        raise ValueError('options cannot contain -o, -f, or -hp, these are '
                         'set automatically')
    port = str(int(port))
    t0 = time.time()
    remote_in = op.join(work_dir, 'temp_%s_raw.fif' % t0)
    remote_out = op.join(work_dir, 'temp_%s_raw_sss.fif' % t0)
    remote_pos = op.join(work_dir, 'temp_%s_raw_sss.pos' % t0)
    print('%sOn %s: copying' % (prefix, host), end='')
    fname_in = op.realpath(fname_in)  # in case it's a symlink
    cmd = ['scp', '-P' + port, fname_in, host + ':' + remote_in]
    run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if fname_pos is not None:
        options += ' -hp ' + remote_pos

    print(', MaxFilter', end='')
    cmd = ['ssh', '-p', port, host,
           'maxfilter -f ' + remote_in + ' -o ' + remote_out + ' ' + options]
    try:
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(', copying to %s' % (op.basename(fname_out),), end='')
        if fname_pos is not None:
            try:
                cmd = ['scp', '-P' + port, host + ':' + remote_pos, fname_pos]
                run_subprocess(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
            except Exception:
                pass
        cmd = ['scp', '-P' + port, host + ':' + remote_out, fname_out]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        print(', cleaning', end='')
        files = [remote_in, remote_out]
        files += [remote_pos] if fname_pos is not None else []
        cmd = ['ssh', '-p', port, host, 'rm -f ' + ' '.join(files)]
        try:
            run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass
        print(' (%i sec)' % (time.time() - t0,))


def run_sss_positions(fname_in, fname_out, host='kasga', opts='', port=22,
                      prefix='  ', work_dir='~/', t_window=None,
                      t_step_min=None, dist_limit=None):
    """Run Maxfilter remotely and fetch resulting file

    Parameters
    ----------
    fname_in : str
        The filename to process. Additional ``-1`` files will be
        automatically detected.
    fname_out : str
        Output filename to use to store the resulting head positions
        on the local machine.
    host : str
        The SSH/scp host to run the command on
    opts : str
        Additional command-line options to pass to MaxFilter.
    port : int
        The SSH port.
    prefix : str
        The prefix to use when printing status updates.
    work_dir : str
        Where to store the temporary files.
    t_window : float | None
        Time window (sec) to use.
    dist_limit : float | None
        Distance limit (m) to use.
    """
    # let's make sure we can actually write where we want
    if not op.isfile(fname_in):
        raise IOError('input file not found: %s' % fname_in)
    if not op.isdir(op.dirname(op.abspath(fname_out))):
        raise IOError('output directory for output file does not exist')
    pout = op.dirname(fname_in)
    fnames_in = [fname_in]
    for ii in range(1, 11):
        next_name = op.splitext(fname_in)[0] + '-%s' % ii + '.fif'
        if op.isfile(next_name):
            fnames_in.append(next_name)
        else:
            break
    if t_window is not None:
        opts += ' -hpiwin %d' % (round(1000 * t_window),)
    if t_step_min is not None:
        opts += ' -hpistep %d' % (round(1000 * t_step_min),)
    if dist_limit is not None:
        opts += ' -hpie %d' % (round(1000 * dist_limit),)

    t0 = time.time()
    print('%sOn %s: copying' % (prefix, host), end='')
    cmd = ['rsync', '--partial', '-Lave', 'ssh -p %s' % port,
           '--include', '*/']
    for fname in fnames_in:
        cmd += ['--include', op.basename(fname)]
    cmd += ['--exclude', '*', op.dirname(fnames_in[0]) + '/',
            '%s:%s' % (host, work_dir)]
    run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    remote_ins = [op.join(work_dir, op.basename(f)) for f in fnames_in]
    fnames_out = [op.basename(r)[:-4] + '.pos' for r in remote_ins]
    for fi, file_out in enumerate(fnames_out):
        remote_out = op.join(work_dir, 'temp_%s_raw_quat.fif' % t0)
        remote_hp = op.join(work_dir, 'temp_%s_hp.txt' % t0)

        print(', running -headpos%s' % opts, end='')
        cmd = ['ssh', '-p', str(port), host,
               '/neuro/bin/util/maxfilter -f ' + remote_ins[fi] + ' -o ' +
               remote_out +
               ' -headpos -format short -hp ' + remote_hp + ' ' + opts]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(', copying', end='')
        cmd = ['scp', '-P' + str(port), host + ':' + remote_hp,
               op.join(pout, file_out)]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cmd = ['ssh', '-p', str(port), host, 'rm -f %s %s %s'
               % (remote_ins[fi], remote_hp, remote_out)]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # concatenate hp pos file for split raw files if any
    data = []
    for f in fnames_out:
        data.append(read_head_pos(op.join(pout, f)))
        os.remove(op.join(pout, f))
    pos_data = np.concatenate(np.array(data))
    print(', writing', end='')
    write_head_pos(fname_out, pos_data)
    print(' (%i sec)' % (time.time() - t0,))


def run_sss_locally(p, subjects, run_indices):
    """Run SSS locally using maxwell filter in python

    See Also
    --------
    mne.preprocessing.maxwell_filter
    """
    data_dir = op.join(op.dirname(__file__), 'data')
    if p.cal_file is None:
        cal_file = op.join(data_dir, 'sss_cal.dat')
    else:
        cal_file = p.cal_file
    if p.ct_file is None:
        ct_file = op.join(data_dir, 'ct_sparse.fif')
    else:
        ct_file = p.ct_file
    assert isinstance(p.tsss_dur, float) and p.tsss_dur > 0
    st_duration = p.tsss_dur
    assert (isinstance(p.sss_regularize, string_types) or
            p.sss_regularize is None)
    reg = p.sss_regularize

    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('    Maxwell filtering subject %g/%g (%s).'
                  % (si + 1, len(subjects), subj))
        # locate raw files with splits
        sss_dir = op.join(p.work_dir, subj, p.sss_dir)
        if not op.isdir(sss_dir):
            os.mkdir(sss_dir)
        raw_files = get_raw_fnames(p, subj, 'raw', erm=False,
                                   run_indices=run_indices[si])
        raw_files_out = get_raw_fnames(p, subj, 'sss', erm=False,
                                       run_indices=run_indices[si])
        erm_files = get_raw_fnames(p, subj, 'raw', 'only')
        erm_files_out = get_raw_fnames(p, subj, 'sss', 'only')
        prebad_file = _prebad(p, subj)

        #  process raw files
        for ii, (r, o) in enumerate(zip(raw_files, raw_files_out)):
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')
            raw = read_raw_fif(r, preload=True, allow_maxshield='yes')
            raw.fix_mag_coil_types()
            _load_meg_bads(raw, prebad_file, disp=ii == 0, prefix=' ' * 6)
            print('      Processing %s ...' % op.basename(r))

            # estimate head position for movement compensation
            head_pos, annot, _ = _head_pos_annot(p, r, prefix='        ')
            try:
                raw.set_annotations(annot)
            except AttributeError:
                raw.annotations = annot

            # get the destination head position
            assert isinstance(p.trans_to, (string_types, tuple, type(None)))
            trans_to = _load_trans_to(p, subj, run_indices[si])

            # filter cHPI signals
            if p.filter_chpi:
                t0 = time.time()
                print('        Filtering cHPI signals ... ', end='')
                raw = filter_chpi(raw)
                print('%i sec' % (time.time() - t0,))

            # apply maxwell filter
            t0 = time.time()
            print('        Running maxwell_filter ... ', end='')
            raw_sss = maxwell_filter(
                raw, origin=p.sss_origin, int_order=p.int_order,
                ext_order=p.ext_order, calibration=cal_file,
                cross_talk=ct_file, st_correlation=p.st_correlation,
                st_duration=st_duration, destination=trans_to,
                coord_frame='head', head_pos=head_pos, regularize=reg,
                bad_condition='warning')
            print('%i sec' % (time.time() - t0,))
            raw_sss.save(o, overwrite=True, buffer_size_sec=None)
        #  process erm files if any
        for ii, (r, o) in enumerate(zip(erm_files, erm_files_out)):
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')
            raw = read_raw_fif(r, preload=True, allow_maxshield='yes')
            raw.fix_mag_coil_types()
            _load_meg_bads(raw, prebad_file, disp=False)
            print('      %s ...' % op.basename(r))
            t0 = time.time()
            print('        Running maxwell_filter ... ', end='')
            # apply maxwell filter
            raw_sss = maxwell_filter(
                raw, int_order=p.int_order, ext_order=p.ext_order,
                calibration=cal_file, cross_talk=ct_file,
                st_correlation=p.st_correlation, st_duration=st_duration,
                destination=None, coord_frame='meg')
            print('%i sec' % (time.time() - t0,))
            raw_sss.save(o, overwrite=True, buffer_size_sec=None)


def _load_trans_to(p, subj, run_indices, raw=None):
    if isinstance(p.trans_to, string_types):
        if p.trans_to == 'median':
            trans_to = op.join(p.work_dir, subj, p.raw_dir,
                               subj + '_median_pos.fif')
            if not op.isfile(trans_to):
                calc_median_hp(p, subj, trans_to, run_indices)
        elif p.trans_to == 'twa':
            trans_to = op.join(p.work_dir, subj, p.raw_dir,
                               subj + '_twa_pos.fif')
            if not op.isfile(trans_to):
                calc_twa_hp(p, subj, trans_to, run_indices)
        else:
            trans_to = p.trans_to
        trans_to = mne.read_trans(trans_to)
    elif p.trans_to is None:
        trans_to = None if raw is None else raw.info['dev_head_t']
    else:
        trans_to = np.array(p.trans_to, float)
        t = np.eye(4)
        if trans_to.shape == (4,):
            theta = np.deg2rad(trans_to[3])
            t[1:3, 1:3] = [[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]]
        elif trans_to.shape != (3,):
            raise ValueError('trans_to must have 3 or 4 elements, '
                             'got shape %s' % (trans_to.shape,))
        t[:3, 3] = trans_to[:3]
        trans_to = mne.Transform('meg', 'head', t)
    if trans_to is not None:
        trans_to = mne.transforms._ensure_trans(trans_to, 'meg', 'head')
    return trans_to


def _load_meg_bads(raw, prebad_file, disp=True, prefix='     '):
    """Helper to load MEG bad channels from a file (pre-MF)"""
    with open(prebad_file, 'r') as fid:
        lines = fid.readlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    if len(lines) > 0:
        try:
            int(lines[0][0])
        except ValueError:
            # MNE-Python type file
            bads = lines
        else:
            # Maxfilter type file
            if len(lines) > 1:
                raise RuntimeError('Could not parse bad file')
            bads = ['MEG%04d' % int(bad) for bad in lines[0].split()]
    else:
        bads = list()
    if disp:
        pl = '' if len(bads) == 1 else 's'
        print('%sMarking %s bad MEG channel%s using %s'
              % (prefix, len(bads), pl, op.basename(prebad_file)))
    raw.info['bads'] = bads
    raw.info._check_consistency()


def extract_expyfun_events(fname, return_offsets=False):
    """Extract expyfun-style serial-coded events from file

    Parameters
    ----------
    fname : str
        Filename to use.
    return_offsets : bool
        If True, return the time of each press relative to trial onset
        in addition to the press number.

    Returns
    -------
    events : array
        Array of events of shape (N, 3), re-coded such that 1 triggers
        are renamed according to their binary expyfun representation.
    presses : list of arrays
        List of all press events that occurred between each one
        trigger. Each array has shape (N_presses,). If return_offset is True,
        then each array has shape (N_presses, 2), with the first column
        as the time offset from the trial trigger.
    orig_events : array
        Original events array.

    Notes
    -----
    When this function translates binary event codes into decimal integers, it
    adds 1 to the value of all events. This is done to prevent the occurrence
    of events with a value of 0 (which downstream processing would treat as
    non-events). If you need to convert the integer event codes back to binary,
    subtract 1 before doing so to yield the original binary values.
    """
    # Read events
    raw = read_raw_fif(fname, allow_maxshield='yes', preload=True)
    raw.pick_types(meg=False, stim=True)
    orig_events = find_events(raw, stim_channel='STI101', shortest_event=0)
    events = list()
    for ch in range(1, 9):
        stim_channel = 'STI%03d' % ch
        ev_101 = find_events(raw, stim_channel='STI101', mask=2 ** (ch - 1),
                             mask_type='and')
        if stim_channel in raw.ch_names:
            ev = find_events(raw, stim_channel=stim_channel)
            if not np.array_equal(ev_101[:, 0], ev[:, 0]):
                warnings.warn('Event coding mismatch between STIM channels')
        else:
            ev = ev_101
        ev[:, 2] = 2 ** (ch - 1)
        events.append(ev)
    events = np.concatenate(events)
    events = events[np.argsort(events[:, 0])]

    # check for the correct number of trials
    aud_idx = np.where(events[:, 2] == 1)[0]
    breaks = np.concatenate(([0], aud_idx, [len(events)]))
    resps = []
    event_nums = []
    for ti in range(len(aud_idx)):
        # pull out responses (they come *after* 1 trig)
        these = events[breaks[ti + 1]:breaks[ti + 2], :]
        resp = these[these[:, 2] > 8]
        resp = np.c_[(resp[:, 0] - events[ti, 0]) / raw.info['sfreq'],
                     np.log2(resp[:, 2]) - 3]
        resps.append(resp if return_offsets else resp[:, 1])

        # look at trial coding, double-check trial type (pre-1 trig)
        these = events[breaks[ti + 0]:breaks[ti + 1], 2]
        serials = these[np.logical_and(these >= 4, these <= 8)]
        en = np.sum(2 ** np.arange(len(serials))[::-1] * (serials == 8)) + 1
        event_nums.append(en)

    these_events = events[aud_idx]
    these_events[:, 2] = event_nums
    return these_events, resps, orig_events


def fix_eeg_files(p, subjects, structurals=None, dates=None, run_indices=None):
    """Reorder EEG channels based on UW cap setup and params

    Reorders only the SSS files based on params, to leave the raw files
    in an unmodified state.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    structurals : list of str
        Subject structural names.
    dates : list of tuple
        Dates that each subject was run.
    run_indices : array-like | None
        Run indices to include.
    """
    if run_indices is None:
        run_indices = [None] * len(subjects)
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Fixing subject %g/%g.' % (si + 1, len(subjects)))
        raw_names = get_raw_fnames(p, subj, 'sss', True, False,
                                   run_indices[si])
        # Now let's make sure we only run files that actually exist
        names = [name for name in raw_names if op.isfile(name)]
        # noinspection PyPep8
        if structurals is not None and structurals[si] is not None and \
                dates is not None:
            assert isinstance(structurals[si], string_types)
            assert dates[si] is None or (isinstance(dates[si], tuple) and
                                         len(dates[si]) == 3)
            assert dates[si] is None or all([isinstance(d, int)
                                             for d in dates[si]])
            anon = dict(first_name=subj, last_name=structurals[si],
                        birthday=dates[si])
        else:
            anon = None
        fix_eeg_channels(names, anon)


def get_fsaverage_medial_vertices(concatenate=True, subjects_dir=None,
                                  vertices=None):
    """Returns fsaverage medial wall vertex numbers

    These refer to the standard fsaverage source space
    (with vertices from 0 to 2*10242-1).

    Parameters
    ----------
    concatenate : bool
        If True, the returned vertices will be indices into the left and right
        hemisphere that are part of the medial wall. This is
        Useful when treating the source space as a single entity (e.g.,
        during clustering).
    subjects_dir : str
        Directory containing subjects data. If None use
        the Freesurfer SUBJECTS_DIR environment variable.
    vertices : None | list
        Can be None to use ``[np.arange(10242)] * 2``.

    Returns
    -------
    vertices : list of array, or array
        The medial wall vertices.
    """
    if vertices is None:
        vertices = [np.arange(10242), np.arange(10242)]
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    label_dir = op.join(subjects_dir, 'fsaverage', 'label')
    lh = read_label(op.join(label_dir, 'lh.Medial_wall.label'))
    rh = read_label(op.join(label_dir, 'rh.Medial_wall.label'))
    if concatenate:
        bad_left = np.where(np.in1d(vertices[0], lh.vertices))[0]
        bad_right = np.where(np.in1d(vertices[1], rh.vertices))[0]
        return np.concatenate((bad_left, bad_right + len(vertices[0])))
    else:
        return [lh.vertices, rh.vertices]


@verbose
def get_fsaverage_label_operator(parc='aparc.a2009s', remove_bads=True,
                                 combine_medial=False, return_labels=False,
                                 subjects_dir=None, verbose=None):
    """Get a label operator matrix for fsaverage."""
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    src = mne.read_source_spaces(op.join(
        subjects_dir, 'fsaverage', 'bem', 'fsaverage-5-src.fif'),
        verbose=False)
    fs_vertices = [np.arange(10242), np.arange(10242)]
    assert all(np.array_equal(a['vertno'], b)
               for a, b in zip(src, fs_vertices))
    labels = mne.read_labels_from_annot('fsaverage', parc)
    # Remove bad labels
    if remove_bads:
        bads = get_fsaverage_medial_vertices(False)
        bads = dict(lh=bads[0], rh=bads[1])
        assert all(b.size > 1 for b in bads.values())
        labels = [label for label in labels
                  if np.in1d(label.vertices, bads[label.hemi]).mean() < 0.8]
        del bads
    if combine_medial:
        labels = combine_medial_labels(labels)
    offsets = dict(lh=0, rh=10242)
    rev_op = np.zeros((20484, len(labels)))
    for li, label in enumerate(labels):
        if isinstance(label, mne.BiHemiLabel):
            use_labels = [label.lh, label.rh]
        else:
            use_labels = [label]
        for ll in use_labels:
            rev_op[ll.get_vertices_used() + offsets[ll.hemi], li:li + 1] = 1.
    # every src vertex is in exactly one label, except medial wall verts
    # assert (rev_op.sum(-1) == 1).sum()
    label_op = mne.SourceEstimate(np.eye(20484), fs_vertices, 0, 1)
    label_op = label_op.extract_label_time_course(labels, src)
    out = (label_op, rev_op)
    if return_labels:
        out += (labels,)
    return out


@verbose
def combine_medial_labels(labels, subject='fsaverage', surf='white',
                          dist_limit=0.02, subjects_dir=None):
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    rrs = dict((hemi, mne.read_surface(op.join(subjects_dir, subject, 'surf',
                                               '%s.%s'
                                               % (hemi, surf)))[0] / 1000.)
               for hemi in ('lh', 'rh'))
    use_labels = list()
    used = np.zeros(len(labels), bool)
    logger.info('Matching medial regions for %s labels on %s %s, d=%0.1f mm'
                % (len(labels), subject, surf, 1000 * dist_limit))
    for li1, l1 in enumerate(labels):
        if used[li1]:
            continue
        used[li1] = True
        use_label = l1.copy()
        rr1 = rrs[l1.hemi][l1.vertices]
        for li2 in np.where(~used)[0]:
            l2 = labels[li2]
            same_name = (l2.name.replace(l2.hemi, '') ==
                         l1.name.replace(l1.hemi, ''))
            if l2.hemi != l1.hemi and same_name:
                rr2 = rrs[l2.hemi][l2.vertices]
                mean_min = np.mean(mne.surface._compute_nearest(
                    rr1, rr2, return_dists=True)[1])
                if mean_min <= dist_limit:
                    use_label += l2
                    used[li2] = True
                    logger.info('  Matched: ' + l1.name)
        use_labels.append(use_label)
    logger.info('Total %d labels' % (len(use_labels),))
    return use_labels


def _restrict_reject_flat(reject, flat, raw):
    """Restrict a reject and flat dict based on channel presence"""
    use_reject, use_flat = dict(), dict()
    for in_, out in zip([reject, flat], [use_reject, use_flat]):
        use_keys = [key for key in in_.keys() if key in raw]
        for key in use_keys:
            out[key] = in_[key]
    return use_reject, use_flat


def save_epochs(p, subjects, in_names, in_numbers, analyses, out_names,
                out_numbers, must_match, decim, run_indices):
    """Generate epochs from raw data based on events

    Can only complete after preprocessing is complete.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    in_names : list of str
        Names of input events.
    in_numbers : list of list of int
        Event numbers (in scored event files) associated with each name.
    analyses : list of str
        Lists of analyses of interest.
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
    decim : int | list of int
        Amount to decimate.
    run_indices : array-like | None
        Run indices to include.
    """
    in_names = np.asanyarray(in_names)
    old_dict = dict()
    for n, e in zip(in_names, in_numbers):
        old_dict[n] = e

    # let's do some sanity checks
    if len(in_names) != len(in_numbers):
        raise RuntimeError('in_names (%d) must have same length as '
                           'in_numbers (%d)'
                           % (len(in_names), len(in_numbers)))
    if np.any(np.array(in_numbers) <= 0):
        raise ValueError('in_numbers must all be > 0')
    if len(out_names) != len(out_numbers):
        raise RuntimeError('out_names must have same length as out_numbers')
    for name, num in zip(out_names, out_numbers):
        num = np.array(num)
        if len(name) != len(np.unique(num[num > 0])):
            raise RuntimeError('each entry in out_names must have length '
                               'equal to the number of unique elements in the '
                               'corresponding entry in out_numbers:\n%s\n%s'
                               % (name, np.unique(num[num > 0])))
        if len(num) != len(in_names):
            raise RuntimeError('each entry in out_numbers must have the same '
                               'length as in_names')
        if (np.array(num) == 0).any():
            raise ValueError('no element of out_numbers can be zero')

    ch_namess = list()
    drop_logs = list()
    sfreqs = set()
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Loading raw files for subject %s.' % subj)
        epochs_dir = op.join(p.work_dir, subj, p.epochs_dir)
        if not op.isdir(epochs_dir):
            os.mkdir(epochs_dir)
        evoked_dir = op.join(p.work_dir, subj, p.inverse_dir)
        if not op.isdir(evoked_dir):
            os.mkdir(evoked_dir)

        # read in raw files
        raw_names = get_raw_fnames(p, subj, 'pca', False, False,
                                   run_indices[si])
        # read in events
        first_samps = []
        last_samps = []
        for raw_fname in raw_names:
            raw = read_raw_fif(raw_fname, preload=False)
            first_samps.append(raw._first_samps[0])
            last_samps.append(raw._last_samps[-1])
        # read in raw files
        raw = [read_raw_fif(fname, preload=False) for fname in raw_names]
        _fix_raw_eog_cals(raw, raw_names)  # EOG epoch scales might be bad!
        raw = concatenate_raws(raw)

        # read in events
        events = list()
        for fname in get_event_fnames(p, subj, run_indices[si]):
            these_events = read_events(fname)
            if len(np.unique(these_events[:, 0])) != len(these_events):
                raise RuntimeError('Non-unique event samples found in %s'
                                   % (fname,))
            events.append(these_events)
        events = concatenate_events(events, first_samps, last_samps)
        if len(np.unique(events[:, 0])) != len(events):
            raise RuntimeError('Non-unique event samples found after '
                               'concatenation')
        # do time adjustment
        t_adj = int(np.round(-p.t_adjust * raw.info['sfreq']))
        events[:, 0] += t_adj
        new_sfreq = raw.info['sfreq'] / decim[si]
        if p.disp_files:
            print('    Epoching data (decim=%s -> sfreq=%s Hz).'
                  % (decim[si], new_sfreq))
        if new_sfreq not in sfreqs:
            if len(sfreqs) > 0:
                warnings.warn('resulting new sampling frequency %s not equal '
                              'to previous values %s' % (new_sfreq, sfreqs))
            sfreqs.add(new_sfreq)
        if p.autoreject_thresholds:
            from autoreject import get_rejection_threshold
            print('     Using autreject to compute rejection thresholds')
            temp_epochs = Epochs(raw, events, event_id=None, tmin=p.tmin,
                                 tmax=p.tmax, baseline=_get_baseline(p),
                                 proj=True, reject=None, flat=None,
                                 preload=True, decim=decim[si])
            new_dict = get_rejection_threshold(temp_epochs)
            use_reject = dict()
            use_reject.update((k, new_dict[k]) for k in p.autoreject_types)
            use_reject, use_flat = _restrict_reject_flat(use_reject,
                                                         p.flat, raw)
        else:
            use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat, raw)

        epochs = Epochs(raw, events, event_id=old_dict, tmin=p.tmin,
                        tmax=p.tmax, baseline=_get_baseline(p),
                        reject=use_reject, flat=use_flat, proj='delayed',
                        preload=True, decim=decim[si], on_missing=p.on_missing,
                        reject_tmin=p.reject_tmin, reject_tmax=p.reject_tmax)
        del raw
        if epochs.events.shape[0] < 1:
            epochs.plot_drop_log()
            raise ValueError('No valid epochs')
        drop_logs.append(epochs.drop_log)
        ch_namess.append(epochs.ch_names)
        # only kept trials that were not dropped
        sfreq = epochs.info['sfreq']
        epochs_fnames, evoked_fnames = get_epochs_evokeds_fnames(p, subj,
                                                                 analyses)
        mat_file, fif_file = epochs_fnames
        # now deal with conditions to save evoked
        if p.disp_files:
            print('    Matching trial counts and saving data to disk.')
        for var, name in ((out_names, 'out_names'),
                          (out_numbers, 'out_numbers'),
                          (must_match, 'must_match'),
                          (evoked_fnames, 'evoked_fnames')):
            if len(var) != len(analyses):
                raise ValueError('len(%s) (%s) != len(analyses) (%s)'
                                 % (name, len(var), len(analyses)))
        for analysis, names, numbers, match, fn in zip(analyses, out_names,
                                                       out_numbers, must_match,
                                                       evoked_fnames):
            # do matching
            numbers = np.asanyarray(numbers)
            nn = numbers[numbers >= 0]
            new_numbers = []
            for num in numbers:
                if num > 0 and num not in new_numbers:
                    # Eventually we could relax this requirement, but not
                    # having it in place is likely to cause people pain...
                    if any(num < n for n in new_numbers):
                        raise RuntimeError('each list of new_numbers must be '
                                           ' monotonically increasing')
                    new_numbers.append(num)
            new_numbers = np.array(new_numbers)
            in_names_match = in_names[match]
            # use some variables to allow safe name re-use
            offset = max(epochs.events[:, 2].max(), new_numbers.max()) + 1
            safety_str = '__mnefun_copy__'
            assert len(new_numbers) == len(names)  # checked above
            if p.match_fun is None:
                # first, equalize trial counts (this will make a copy)
                e = epochs[list(in_names[numbers > 0])]
                if len(in_names_match) > 1:
                    e.equalize_event_counts(in_names_match)

                # second, collapse relevant types
                for num, name in zip(new_numbers, names):
                    collapse = [x for x in in_names[num == numbers]
                                if x in e.event_id]
                    combine_event_ids(e, collapse,
                                      {name + safety_str: num + offset},
                                      copy=False)
                for num, name in zip(new_numbers, names):
                    e.events[e.events[:, 2] == num + offset, 2] -= offset
                    e.event_id[name] = num
                    del e.event_id[name + safety_str]
            else:  # custom matching
                e = p.match_fun(epochs.copy(), analysis, nn,
                                in_names_match, names)

            # now make evoked for each out type
            evokeds = list()
            for name in names:
                this_e = e[name]
                if len(this_e) > 0:
                    evokeds.append(this_e.average())
                    evokeds.append(this_e.standard_error())
            write_evokeds(fn, evokeds)
            naves = [str(n) for n in sorted(set([evoked.nave
                                                 for evoked in evokeds]))]
            naves = ', '.join(naves)
            if p.disp_files:
                print('      Analysis "%s": %s epochs / condition'
                      % (analysis, naves))

        if p.disp_files:
            print('    Saving epochs to disk.')
        if 'mat' in p.epochs_type:
            spio.savemat(mat_file, dict(epochs=epochs.get_data(),
                                        events=epochs.events, sfreq=sfreq,
                                        drop_log=epochs.drop_log),
                         do_compression=True, oned_as='column')
        if 'fif' in p.epochs_type:
            epochs.save(fif_file)

    if p.plot_drop_logs:
        for subj, drop_log in zip(subjects, drop_logs):
            plot_drop_log(drop_log, threshold=p.drop_thresh, subject=subj)


def gen_inverses(p, subjects, run_indices):
    """Generate inverses

    Can only complete successfully following forward solution
    calculation and covariance estimation.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    run_indices : array-like | None
        Run indices to include.
    """
    for si, subj in enumerate(subjects):
        out_flags, meg_bools, eeg_bools = [], [], []
        if p.disp_files:
            print('  Subject %s' % subj, end='')
        inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        if not op.isdir(inv_dir):
            os.mkdir(inv_dir)
        make_erm_inv = len(p.runs_empty) > 0

        # Shouldn't matter which raw file we use
        raw_fname = get_raw_fnames(p, subj, 'pca', True, False,
                                   run_indices[si])[0]
        raw = read_raw_fif(raw_fname)
        meg, eeg = 'meg' in raw, 'eeg' in raw

        if meg:
            out_flags += ['-meg']
            meg_bools += [True]
            eeg_bools += [False]
        if eeg:
            out_flags += ['-eeg']
            meg_bools += [False]
            eeg_bools += [True]
        if meg and eeg:
            out_flags += ['-meg-eeg']
            meg_bools += [True]
            eeg_bools += [True]
        if p.compute_rank:
            from mne.utils import estimate_rank
            epochs_fnames, _ = get_epochs_evokeds_fnames(p, subj, p.analyses)
            _, fif_file = epochs_fnames
            epochs = mne.read_epochs(fif_file)
            rank = dict()
            if meg:
                eps = epochs.copy().pick_types(meg=meg, eeg=False)
                eps = eps.get_data().transpose([1, 0, 2])
                eps = eps.reshape(len(eps), -1)
                rank['meg'] = estimate_rank(eps, tol=1e-6)
            if eeg:
                eps = epochs.copy().pick_types(meg=False, eeg=eeg)
                eps = eps.get_data().transpose([1, 0, 2])
                eps = eps.reshape(len(eps), -1)
                rank['eeg'] = estimate_rank(eps, tol=1e-6)
            for k, v in rank.items():
                print(' : %s rank %2d' % (k.upper(), v), end='')
        else:
            rank = None
        if make_erm_inv:
            erm_name = op.join(cov_dir, safe_inserter(p.runs_empty[0], subj) +
                               p.pca_extra + p.inv_tag + '-cov.fif')
            empty_cov = read_cov(erm_name)
            if empty_cov.get('method', 'empirical') == 'empirical':
                empty_cov = regularize(empty_cov, raw.info)
        for name in p.inv_names:
            s_name = safe_inserter(name, subj)
            temp_name = s_name + ('-%d' % p.lp_cut) + p.inv_tag
            fwd_name = op.join(fwd_dir, s_name + p.inv_tag + '-fwd.fif')
            fwd = read_forward_solution(fwd_name)
            fwd = convert_forward_solution(fwd, surf_ori=True)
            looses = [1]
            tags = [p.inv_free_tag]
            fixeds = [False]
            depths = [0.8]
            if fwd['src'][0]['type'] == 'surf':
                looses += [0, 0.2]
                tags += [p.inv_fixed_tag, p.inv_loose_tag]
                fixeds += [True, False]
                depths += [0.8, 0.8]
            cov_name = op.join(cov_dir, safe_inserter(name, subj) +
                               ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
            cov = read_cov(cov_name)
            if cov.get('method', 'empirical') == 'empirical':
                cov = regularize(cov, raw.info)
            for f, m, e in zip(out_flags, meg_bools, eeg_bools):
                fwd_restricted = pick_types_forward(fwd, meg=m, eeg=e)
                for l, s, x, d in zip(looses, tags, fixeds, depths):
                    inv_name = op.join(inv_dir, temp_name + f + s + '-inv.fif')
                    inv = make_inverse_operator(raw.info, fwd_restricted, cov,
                                                loose=l, depth=d, fixed=x,
                                                use_cps=True, rank=rank)
                    write_inverse_operator(inv_name, inv)
                    if (not e) and make_erm_inv:
                        inv_name = op.join(inv_dir, temp_name + f +
                                           p.inv_erm_tag + s + '-inv.fif')
                        inv = make_inverse_operator(raw.info, fwd_restricted,
                                                    empty_cov, loose=l,
                                                    depth=d, fixed=x)
                        write_inverse_operator(inv_name, inv)


def gen_forwards(p, subjects, structurals, run_indices):
    """Generate forward solutions

    Can only complete successfully once coregistration is performed
    (usually in mne_analyze).

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    structurals : list (of str or None)
        The structural data names for each subject (e.g., ['AKCLEE_101', ...]).
        If None, a spherical BEM and volume grid space will be used.
    run_indices : array-like | None
        Run indices to include.
    """
    for si, subj in enumerate(subjects):
        struc = structurals[si]
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        if not op.isdir(fwd_dir):
            os.mkdir(fwd_dir)
        raw_fname = get_raw_fnames(p, subj, 'sss', False, False,
                                   run_indices[si])[0]
        info = read_info(raw_fname)
        bem, src, trans, bem_type = _get_bem_src_trans(p, info, subj, struc)
        if not getattr(p, 'translate_positions', True):
            raise RuntimeError('Not translating positions is no longer '
                               'supported')
        print('  Creating forward solution(s) using a %s for %s...'
              % (bem_type, subj))
        # XXX Don't actually need to generate a different fwd for each inv
        # anymore, since all runs are included, but changing the filename
        # would break a lot of existing pipelines :(
        for ii, (inv_name, inv_run) in enumerate(zip(p.inv_names,
                                                     p.inv_runs)):
            fwd_name = op.join(fwd_dir, safe_inserter(inv_name, subj) +
                               p.inv_tag + '-fwd.fif')
            fwd = make_forward_solution(
                info, trans, src, bem, n_jobs=p.n_jobs, mindist=p.fwd_mindist)
            write_forward_solution(fwd_name, fwd, overwrite=True)


def _get_bem_src_trans(p, info, subj, struc):
    subjects_dir = mne.utils.get_subjects_dir(p.subjects_dir, raise_error=True)
    assert isinstance(subjects_dir, string_types)
    if struc is None:  # spherical case
        bem, src, trans = _spherical_conductor(info, subj, p.src_pos)
        bem_type = 'spherical-model'
    else:
        trans = op.join(p.work_dir, subj, p.trans_dir, subj + '-trans.fif')
        if not op.isfile(trans):
            old = trans
            trans = op.join(p.work_dir, subj, p.trans_dir,
                            subj + '-trans_head2mri.txt')
            if not op.isfile(trans):
                raise IOError('Unable to find head<->MRI trans files in:\n'
                              '%s\n%s' % (old, trans))
        trans = mne.read_trans(trans)
        trans = mne.transforms._ensure_trans(trans, 'mri', 'head')
        for mid in ('oct6', 'oct-6'):
            src_space_file = op.join(subjects_dir, struc, 'bem',
                                     '%s-%s-src.fif' % (struc, mid))
            if op.isfile(src_space_file):
                break
        else:  # if neither exists, use last filename
            print('  Creating source space for %s...' % subj)
            src = setup_source_space(
                struc, spacing='oct6', subjects_dir=p.subjects_dir,
                n_jobs=p.n_jobs)
            write_source_spaces(src_space_file, src)
        src = read_source_spaces(src_space_file)
        bem = op.join(subjects_dir, struc, 'bem', '%s-%s-bem-sol.fif'
                      % (struc, p.bem_type))
        bem = mne.read_bem_solution(bem, verbose=False)
        bem_type = ('%s-layer BEM' % len(bem['surfs']))
    return bem, src, trans, bem_type


def gen_covariances(p, subjects, run_indices):
    """Generate forward solutions

    Can only complete successfully once preprocessing is performed.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    run_indices : array-like | None
        Run indices to include.
    """
    for si, subj in enumerate(subjects):
        print('  Subject %s/%s...' % (si + 1, len(subjects)))
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        if not op.isdir(cov_dir):
            os.mkdir(cov_dir)

        # Make empty room cov
        if p.runs_empty:
            if len(p.runs_empty) > 1:
                raise ValueError('Too many empty rooms; undefined output!')
            new_run = safe_inserter(p.runs_empty[0], subj)
            empty_cov_name = op.join(cov_dir, new_run + p.pca_extra +
                                     p.inv_tag + '-cov.fif')
            empty_fif = get_raw_fnames(p, subj, 'pca', 'only', False)[0]
            raw = read_raw_fif(empty_fif, preload=True)
            raw.pick_types(meg=True, eog=True, exclude='bads')
            use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat, raw)
            cov = compute_raw_covariance(raw, reject=use_reject, flat=use_flat)
            write_cov(empty_cov_name, cov)

        # Make evoked covariances
        for ii, (inv_name, inv_run) in enumerate(zip(p.inv_names, p.inv_runs)):
            cov_name = op.join(cov_dir, safe_inserter(inv_name, subj) +
                               ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
            if run_indices[si] is None:
                ridx = inv_run
            else:
                ridx = np.intersect1d(run_indices[si], inv_run)
            raw_fnames = get_raw_fnames(p, subj, 'pca', False, False, ridx)
            eve_fnames = get_event_fnames(p, subj, ridx)

            raws = []
            first_samps = []
            last_samps = []
            for raw_fname in raw_fnames:
                raws.append(read_raw_fif(raw_fname, preload=False))
                first_samps.append(raws[-1]._first_samps[0])
                last_samps.append(raws[-1]._last_samps[-1])
            _fix_raw_eog_cals(raws, raw_fnames)  # safe b/c cov only needs MEEG
            raw = concatenate_raws(raws)
            events = [read_events(e) for e in eve_fnames]
            if p.pick_events_cov is not None:
                old_count = sum(len(e) for e in events)
                if callable(p.pick_events_cov):
                    picker = p.pick_events_cov
                else:
                    picker = p.pick_events_cov[ii]
                events = [picker(e) for e in events]
                new_count = sum(len(e) for e in events)
                print('  Using %s/%s events for %s'
                      % (new_count, old_count, op.basename(cov_name)))
            events = concatenate_events(events, first_samps,
                                        last_samps)
            use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat, raw)
            epochs = Epochs(raw, events, event_id=None, tmin=p.bmin,
                            tmax=p.bmax, baseline=(None, None), proj=False,
                            reject=use_reject, flat=use_flat, preload=True)
            epochs.pick_types(meg=True, eeg=True, exclude=[])
            cov = compute_covariance(epochs, method=p.cov_method)
            write_cov(cov_name, cov)


def _fix_raw_eog_cals(raws, raw_names):
    """Fix for annoying issue where EOG cals don't match"""
    # Warning: this will only produce correct EOG scalings with preloaded
    # raw data!
    picks = pick_types(raws[0].info, eeg=False, meg=False, eog=True,
                       exclude=[])
    if len(picks) > 0:
        first_cals = _cals(raws[0])[picks]
        for ri, r in enumerate(raws[1:]):
            picks_2 = pick_types(r.info, eeg=False, meg=False, eog=True,
                                 exclude=[])
            assert np.array_equal(picks, picks_2)
            these_cals = _cals(r)[picks]
            if not np.array_equal(first_cals, these_cals):
                warnings.warn('Adjusting EOG cals for %s' % raw_names[ri + 1])
                _cals(r)[picks] = first_cals


def _cals(raw):
    """Helper to deal with the .cals->._cals attribute change"""
    try:
        return raw._cals
    except AttributeError:
        return raw.cals


def _get_fir_kwargs(fir_design):
    """Get FIR kwargs in backward-compatible way."""
    fir_kwargs = dict()
    old_kwargs = dict()
    if 'fir_design' in get_args(mne.filter.filter_data):
        fir_kwargs.update(fir_design=fir_design)
        old_kwargs.update(fir_design='firwin2')
    elif fir_design != 'firwin2':
        raise RuntimeError('cannot use fir_design=%s with old MNE'
                           % fir_design)
    return fir_kwargs, old_kwargs


# noinspection PyPep8Naming
def _raw_LRFCP(raw_names, sfreq, l_freq, h_freq, n_jobs, n_jobs_resample,
               projs, bad_file, disp_files=False, method='fir',
               filter_length=32768, apply_proj=True, preload=True,
               force_bads=False, l_trans=0.5, h_trans=0.5,
               allow_maxshield=False, phase='zero-double', fir_window='hann',
               fir_design='firwin2', pick=True):
    """Helper to load, filter, concatenate, then project raw files"""
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    if disp_files:
        print('    Loading and filtering %d files.' % len(raw_names))
    raw = list()
    for rn in raw_names:
        r = read_raw_fif(rn, preload=True, allow_maxshield='yes')
        if pick:
            r.pick_types(meg=True, eeg=True, eog=True, ecg=True, exclude=())
        r.load_bad_channels(bad_file, force=force_bads)
        r.pick_types(meg=True, eeg=True, eog=True, ecg=True, exclude=[])
        if _needs_eeg_average_ref_proj(r.info):
            r.set_eeg_reference(projection=True)
        if sfreq is not None:
            r.resample(sfreq, n_jobs=n_jobs_resample, npad='auto')
        fir_kwargs = _get_fir_kwargs(fir_design)[0]
        if l_freq is not None or h_freq is not None:
            r.filter(l_freq=l_freq, h_freq=h_freq, picks=None,
                     n_jobs=n_jobs, method=method,
                     filter_length=filter_length, phase=phase,
                     l_trans_bandwidth=l_trans, h_trans_bandwidth=h_trans,
                     fir_window=fir_window, **fir_kwargs)
        raw.append(r)
    _fix_raw_eog_cals(raw, raw_names)
    raws_del = raw[1:]

    raw = concatenate_raws(raw, preload=preload)
    for r in raws_del:
        del r
    if disp_files and apply_proj and len(projs) > 0:
        print('    Adding and applying projectors.')
    raw.add_proj(projs)
    if apply_proj:
        raw.apply_proj()
    return raw


def do_preprocessing_combined(p, subjects, run_indices):
    """Do preprocessing on all raw files together

    Calculates projection vectors to use to clean data.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    run_indices : array-like | None
        Run indices to include.
    """
    drop_logs = list()
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Preprocessing subject %g/%g (%s).'
                  % (si + 1, len(subjects), subj))
        pca_dir = op.join(p.work_dir, subj, p.pca_dir)
        bad_dir = op.join(p.work_dir, subj, p.bad_dir)

        # Create SSP projection vectors after marking bad channels
        raw_names = get_raw_fnames(p, subj, 'sss', False, False,
                                   run_indices[si])
        empty_names = get_raw_fnames(p, subj, 'sss', 'only')
        for r in raw_names + empty_names:
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')

        bad_file = op.join(bad_dir, 'bad_ch_' + subj + p.bad_tag)
        fir_kwargs, old_kwargs = _get_fir_kwargs(p.fir_design)
        if isinstance(p.auto_bad, float):
            print('    Creating bad channel file, marking bad channels:\n'
                  '        %s' % bad_file)
            if not op.isdir(bad_dir):
                os.mkdir(bad_dir)
            # do autobad
            raw = _raw_LRFCP(raw_names, p.proj_sfreq, None, None, p.n_jobs_fir,
                             p.n_jobs_resample, list(), None, p.disp_files,
                             method='fir', filter_length=p.filter_length,
                             apply_proj=False, force_bads=False,
                             l_trans=p.hp_trans, h_trans=p.lp_trans,
                             phase=p.phase, fir_window=p.fir_window,
                             pick=True, **fir_kwargs)
            events = fixed_len_events(p, raw)
            # do not mark eog channels bad
            meg, eeg = 'meg' in raw, 'eeg' in raw
            picks = pick_types(raw.info, meg=meg, eeg=eeg, eog=False,
                               exclude=[])
            assert p.auto_bad_flat is None or isinstance(p.auto_bad_flat, dict)
            assert p.auto_bad_reject is None or isinstance(p.auto_bad_reject,
                                                           dict)
            if p.auto_bad_reject is None and p.auto_bad_flat is None:
                raise RuntimeError('Auto bad channel detection active. Noisy '
                                   'and flat channel detection '
                                   'parameters not defined. '
                                   'At least one criterion must be defined.')
            epochs = Epochs(raw, events, None, p.tmin, p.tmax,
                            baseline=_get_baseline(p), picks=picks,
                            reject=p.auto_bad_reject, flat=p.auto_bad_flat,
                            proj=True, preload=True, decim=1,
                            reject_tmin=p.reject_tmin,
                            reject_tmax=p.reject_tmax)
            # channel scores from drop log
            scores = Counter([ch for d in epochs.drop_log for ch in d])
            ch_names = np.array(list(scores.keys()))
            # channel scores expressed as percentile and rank ordered
            counts = (100 * np.array([scores[ch] for ch in ch_names], float) /
                      len(epochs.drop_log))
            order = np.argsort(counts)[::-1]
            # boolean array masking out channels with <= % epochs dropped
            mask = counts[order] > p.auto_bad
            badchs = ch_names[order[mask]]
            if len(badchs) > 0:
                # Make sure we didn't get too many bad MEG or EEG channels
                for m, e, thresh in zip([True, False], [False, True],
                                        [p.auto_bad_meg_thresh,
                                         p.auto_bad_eeg_thresh]):
                    picks = pick_types(epochs.info, meg=m, eeg=e, exclude=[])
                    if len(picks) > 0:
                        ch_names = [epochs.ch_names[pp] for pp in picks]
                        n_bad_type = sum(ch in ch_names for ch in badchs)
                        if n_bad_type > thresh:
                            stype = 'meg' if m else 'eeg'
                            raise RuntimeError('Too many bad %s channels '
                                               'found: %s > %s'
                                               % (stype, n_bad_type, thresh))

                print('    The following channels resulted in greater than '
                      '{:.0f}% trials dropped:\n'.format(p.auto_bad * 100))
                print(badchs)
                with open(bad_file, 'w') as f:
                    f.write('\n'.join(badchs))
        if not op.isfile(bad_file):
            print('    No bad channel file found, clearing bad channels:\n'
                  '        %s' % bad_file)
            bad_file = None

        proj_nums = p.proj_nums
        eog_t_lims = [-0.25, 0.25]
        ecg_t_lims = [-0.08, 0.08]
        eog_f_lims = [0, 2]
        ecg_f_lims = [5, 35]

        ecg_eve = op.join(pca_dir, 'preproc_ecg-eve.fif')
        ecg_proj = op.join(pca_dir, 'preproc_ecg-proj.fif')
        eog_eve = op.join(pca_dir, 'preproc_blink-eve.fif')
        eog_proj = op.join(pca_dir, 'preproc_blink-proj.fif')
        cont_proj = op.join(pca_dir, 'preproc_cont-proj.fif')
        all_proj = op.join(pca_dir, 'preproc_all-proj.fif')

        if not op.isdir(pca_dir):
            os.mkdir(pca_dir)

        pre_list = [r for ri, r in enumerate(raw_names)
                    if ri in p.get_projs_from]

        # Calculate and apply continuous projectors if requested
        projs = list()
        raw_orig = _raw_LRFCP(
            raw_names=pre_list, sfreq=p.proj_sfreq, l_freq=None, h_freq=None,
            n_jobs=p.n_jobs_fir, n_jobs_resample=p.n_jobs_resample,
            projs=projs, bad_file=bad_file, disp_files=p.disp_files,
            method='fir', filter_length=p.filter_length, force_bads=False,
            l_trans=p.hp_trans, h_trans=p.lp_trans, phase=p.phase,
            fir_window=p.fir_window, pick=True, **fir_kwargs)

        # Apply any user-supplied extra projectors
        if p.proj_extra is not None:
            if p.disp_files:
                print('    Adding extra projectors from "%s".' % p.proj_extra)
            extra_proj = op.join(pca_dir, p.proj_extra)
            projs = read_proj(extra_proj)

        # Calculate and apply ERM projectors
        proj_nums = np.array(proj_nums, int)
        if proj_nums.shape != (3, 3):
            raise ValueError('proj_nums must be an array with shape (3, 3), '
                             'got %s' % (projs.shape,))
        if any(proj_nums[2]):
            if len(empty_names) >= 1:
                if p.disp_files:
                    print('    Computing continuous projectors using ERM.')
                # Use empty room(s), but processed the same way
                raw = _raw_LRFCP(
                    raw_names=empty_names, sfreq=p.proj_sfreq,
                    l_freq=None, h_freq=None, n_jobs=p.n_jobs_fir,
                    n_jobs_resample=p.n_jobs_resample, projs=projs,
                    bad_file=bad_file, disp_files=p.disp_files, method='fir',
                    filter_length=p.filter_length, force_bads=True,
                    l_trans=p.hp_trans, h_trans=p.lp_trans,
                    phase=p.phase, fir_window=p.fir_window, **fir_kwargs)
            else:
                if p.disp_files:
                    print('    Computing continuous projectors using data.')
                raw = raw_orig.copy()
            raw.filter(None, p.cont_lp, n_jobs=p.n_jobs_fir, method='fir',
                       filter_length=p.filter_length, h_trans_bandwidth=0.5,
                       fir_window=p.fir_window, phase=p.phase, **fir_kwargs)
            raw.add_proj(projs)
            raw.apply_proj()
            pr = compute_proj_raw(raw, duration=1, n_grad=proj_nums[2][0],
                                  n_mag=proj_nums[2][1], n_eeg=proj_nums[2][2],
                                  reject=None, flat=None, n_jobs=p.n_jobs_mkl)
            write_proj(cont_proj, pr)
            projs.extend(pr)
            del raw

        # Calculate and apply the ECG projectors
        if any(proj_nums[0]):
            if p.disp_files:
                print('    Computing ECG projectors.')
            raw = raw_orig.copy()

            raw.filter(ecg_f_lims[0], ecg_f_lims[1], n_jobs=p.n_jobs_fir,
                       method='fir', filter_length=p.filter_length,
                       l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                       phase='zero-double', fir_window='hann',
                       **old_kwargs)
            raw.add_proj(projs)
            raw.apply_proj()
            pr, ecg_events, drop_log = \
                compute_proj_ecg(raw, n_grad=proj_nums[0][0],
                                 n_jobs=p.n_jobs_mkl,
                                 n_mag=proj_nums[0][1], n_eeg=proj_nums[0][2],
                                 tmin=ecg_t_lims[0], tmax=ecg_t_lims[1],
                                 l_freq=None, h_freq=None, no_proj=True,
                                 qrs_threshold='auto', ch_name=p.ecg_channel,
                                 reject=p.ssp_ecg_reject, return_drop_log=True,
                                 average=p.proj_ave)
            n_good = sum(len(d) == 0 for d in drop_log)
            if n_good >= 20:
                write_events(ecg_eve, ecg_events)
                write_proj(ecg_proj, pr)
                projs.extend(pr)
            else:
                plot_drop_log(drop_log)
                raw.plot(events=ecg_events)
                raise RuntimeError('Only %d/%d good ECG epochs found'
                                   % (n_good, len(ecg_events)))
            del raw

        # Next calculate and apply the EOG projectors
        if any(proj_nums[1]):
            if p.disp_files:
                print('    Computing EOG projectors.')
            raw = raw_orig.copy()
            raw.filter(eog_f_lims[0], eog_f_lims[1], n_jobs=p.n_jobs_fir,
                       method='fir', filter_length=p.filter_length,
                       l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                       phase='zero-double', fir_window='hann',
                       **old_kwargs)
            raw.add_proj(projs)
            raw.apply_proj()
            pr, eog_events = \
                compute_proj_eog(raw, n_grad=proj_nums[1][0],
                                 n_jobs=p.n_jobs_mkl,
                                 n_mag=proj_nums[1][1], n_eeg=proj_nums[1][2],
                                 tmin=eog_t_lims[0], tmax=eog_t_lims[1],
                                 l_freq=None, h_freq=None, no_proj=True,
                                 ch_name=p.eog_channel,
                                 reject=p.ssp_eog_reject, average=p.proj_ave)
            if eog_events.shape[0] >= 5:
                write_events(eog_eve, eog_events)
                write_proj(eog_proj, pr)
                projs.extend(pr)
            else:
                warnings.warn('Only %d EOG events!' % eog_events.shape[0])
            del raw

        # save the projectors
        write_proj(all_proj, projs)

        # look at raw_orig for trial DQs now, it will be quick
        raw_orig.filter(p.hp_cut, p.lp_cut, n_jobs=p.n_jobs_fir, method='fir',
                        filter_length=p.filter_length,
                        l_trans_bandwidth=p.hp_trans, phase=p.phase,
                        h_trans_bandwidth=p.lp_trans, fir_window=p.fir_window,
                        **fir_kwargs)
        raw_orig.add_proj(projs)
        raw_orig.apply_proj()
        # now let's epoch with 1-sec windows to look for DQs
        events = fixed_len_events(p, raw_orig)
        use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat,
                                                     raw_orig)
        epochs = Epochs(raw_orig, events, None, p.tmin, p.tmax, preload=False,
                        baseline=_get_baseline(p), reject=use_reject,
                        flat=use_flat, proj=True)
        try:
            epochs.drop_bad()
        except AttributeError:  # old way
            epochs.drop_bad_epochs()
        drop_logs.append(epochs.drop_log)
        del raw_orig
        del epochs
    if p.plot_drop_logs:
        for subj, drop_log in zip(subjects, drop_logs):
            plot_drop_log(drop_log, p.drop_thresh, subject=subj)


def apply_preprocessing_combined(p, subjects, run_indices):
    """Actually apply and save the preprocessing (projs, filtering)

    Can only run after do_preprocessing_combined is done.
    Filters data, adds projection vectors, and saves to disk
    (overwriting old files).

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    run_indices : array-like | None
        Run indices to include.
    """
    # Now actually save some data
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Applying processing to subject %g/%g.'
                  % (si + 1, len(subjects)))
        pca_dir = op.join(p.work_dir, subj, p.pca_dir)
        names_in = get_raw_fnames(p, subj, 'sss', False, False,
                                  run_indices[si])
        names_out = get_raw_fnames(p, subj, 'pca', False, False,
                                   run_indices[si])
        erm_in = get_raw_fnames(p, subj, 'sss', 'only')
        erm_out = get_raw_fnames(p, subj, 'pca', 'only')
        bad_dir = op.join(p.work_dir, subj, p.bad_dir)
        bad_file = op.join(bad_dir, 'bad_ch_' + subj + p.bad_tag)
        bad_file = None if not op.isfile(bad_file) else bad_file
        all_proj = op.join(pca_dir, 'preproc_all-proj.fif')
        projs = read_proj(all_proj)
        fir_kwargs = _get_fir_kwargs(p.fir_design)[0]
        if len(erm_in) > 0:
            for ii, (r, o) in enumerate(zip(erm_in, erm_out)):
                if p.disp_files:
                    print('    Processing erm file %d/%d.'
                          % (ii + 1, len(erm_in)))
            raw = _raw_LRFCP(
                raw_names=r, sfreq=None, l_freq=p.hp_cut, h_freq=p.lp_cut,
                n_jobs=p.n_jobs_fir, n_jobs_resample=p.n_jobs_resample,
                projs=projs, bad_file=bad_file, disp_files=False, method='fir',
                apply_proj=False, filter_length=p.filter_length,
                force_bads=True, l_trans=p.hp_trans, h_trans=p.lp_trans,
                phase=p.phase, fir_window=p.fir_window, pick=False,
                **fir_kwargs)
            raw.save(o, overwrite=True, buffer_size_sec=None)
        for ii, (r, o) in enumerate(zip(names_in, names_out)):
            if p.disp_files:
                print('    Processing file %d/%d.'
                      % (ii + 1, len(names_in)))
            raw = _raw_LRFCP(
                raw_names=r, sfreq=None, l_freq=p.hp_cut, h_freq=p.lp_cut,
                n_jobs=p.n_jobs_fir, n_jobs_resample=p.n_jobs_resample,
                projs=projs, bad_file=bad_file, disp_files=False, method='fir',
                apply_proj=False, filter_length=p.filter_length,
                force_bads=False, l_trans=p.hp_trans, h_trans=p.lp_trans,
                phase=p.phase, fir_window=p.fir_window, pick=False,
                **fir_kwargs)
            raw.save(o, overwrite=True, buffer_size_sec=None)
        # look at raw_clean for ExG events
        if p.plot_raw:
            _viz_raw_ssp_events(p, subj, run_indices[si])


class FakeEpochs(object):
    """Make iterable epoch-like class, convenient for MATLAB transition"""

    def __init__(self, data, ch_names, tmin=-0.2, sfreq=1000.0):
        raise RuntimeError('Use mne.EpochsArray instead')


def timestring(t):
    """Reformat time to convenient string

    Parameters
    ----------
    t : float
        Elapsed time in seconds.

    Returns
    time : str
        The time in HH:MM:SS.
    """

    def rediv(ll, b):
        return list(divmod(ll[0], b)) + ll[1:]

    return "%d:%02d:%02d.%03d" % tuple(reduce(rediv, [[t * 1000, ], 1000, 60,
                                                      60]))


def source_script(script_name):
    """Set environmental variables by source-ing a bash script

    Parameters
    ----------
    script_name : str
        Path to the script to execute and get the environment variables from.
    """
    cmd = ['bash', '-c', 'source ' + script_name + ' > /dev/null && env']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in proc.stdout:
        (key, _, value) = line.partition("=")
        os.environ[key] = value.strip()
    proc.communicate()


def fixed_len_events(p, raw):
    """Create fixed length trial events from raw object"""
    dur = p.tmax - p.tmin
    events = make_fixed_length_events(raw, 1, duration=dur)
    return events


def _viz_raw_ssp_events(p, subj, ridx):
    """Helper to plot filtered cleaned raw trace with ExG events"""
    pca_dir = op.join(p.work_dir, subj, p.pca_dir)
    raw_names = get_raw_fnames(p, subj, 'sss', False, False, ridx)
    pre_list = [r for ri, r in enumerate(raw_names)
                if ri in p.get_projs_from]
    all_proj = op.join(pca_dir, 'preproc_all-proj.fif')
    projs = read_proj(all_proj)
    colors = dict()
    ev = np.zeros((0, 3), int)
    for n, c, cid in zip(['ecg', 'blink'], ['r', 'b'], [999, 998]):
        fname = op.join(pca_dir, 'preproc_%s-eve.fif' % n)
        if op.isfile(fname):
            ev = np.concatenate((ev, read_events(fname)))
            colors[cid] = c
    ev = ev[np.argsort(ev[:, 0], axis=0)]
    raw = _raw_LRFCP(pre_list, p.proj_sfreq, None, None, p.n_jobs_fir,
                     p.n_jobs_resample, projs, None, p.disp_files,
                     method='fir', filter_length=p.filter_length,
                     force_bads=False, l_trans=p.hp_trans, h_trans=p.lp_trans)
    raw.plot(events=ev, event_color=colors)


def _prebad(p, subj):
    """Helper for locating file containing bad channels during acq"""
    prebad_file = op.join(p.work_dir, subj, p.raw_dir, subj + '_prebad.txt')
    if not op.isfile(prebad_file):  # SSS prebad file
        raise RuntimeError('Could not find SSS prebad file: %s' % prebad_file)
    return prebad_file


def _head_pos_annot(p, raw_fname, prefix='  '):
    """Locate head position estimation file and do annotations."""
    if p.movecomp is None:
        return None, None, None
    t_window = p.coil_t_window
    raw = mne.io.read_raw_fif(raw_fname, allow_maxshield='yes')
    if t_window == 'auto':
        hpi_freqs, _, _ = _get_hpi_info(raw.info)
        # Use the longer of 5 cycles and the difference in HPI freqs.
        # This will be 143 ms for 7 Hz spacing (old) and
        # 60 ms for 83 Hz lowest freq.
        t_window = max(5. / min(hpi_freqs), 1. / np.diff(hpi_freqs).min())
        t_window = round(1000 * t_window) / 1000.  # round to ms
    pos_fname = raw_fname[:-4] + '.pos'
    if not op.isfile(pos_fname):
        # XXX Someday we can do:
        # head_pos = _calculate_chpi_positions(
        #     raw, t_window=t_window, dist_limit=dist_limit)
        # write_head_positions(pos_fname, head_pos)
        print('%sEstimating position file %s' % (prefix, pos_fname,))
        run_sss_positions(raw_fname, pos_fname,
                          host=p.sws_ssh, port=p.sws_port, prefix=prefix,
                          work_dir=p.sws_dir, t_window=t_window,
                          t_step_min=p.coil_t_step_min,
                          dist_limit=p.coil_dist_limit)
    head_pos = read_head_pos(pos_fname)

    # do the coil counts
    count_fname = raw_fname[:-4] + '-counts.h5'
    if p.coil_dist_limit is None or p.coil_bad_count_duration_limit is None:
        fit_data = None
    else:
        if not op.isfile(count_fname):
            fit_t, counts, n_coils = compute_good_coils(
                raw, p.coil_t_step_min, t_window, p.coil_dist_limit,
                prefix=prefix, verbose=True)
            write_hdf5(count_fname,
                       dict(fit_t=fit_t, counts=counts, n_coils=n_coils,
                            t_step=p.coil_t_step_min, t_window=t_window,
                            coil_dist_limit=p.coil_dist_limit), title='mnefun')
        fit_data = read_hdf5(count_fname, 'mnefun')
        for key, val in (('t_step', p.coil_t_step_min),
                         ('t_window', t_window),
                         ('coil_dist_limit', p.coil_dist_limit)):
            if fit_data[key] != val:
                raise RuntimeError('Data mismatch %s (%s != %s), set '
                                   'to match existing file or delete it:\n%s'
                                   % (key, val, fit_data[key], count_fname))

    # do the annotations
    lims = [p.rotation_limit, p.translation_limit, p.coil_dist_limit,
            p.coil_t_step_min, t_window, p.coil_bad_count_duration_limit]
    annot_fname = raw_fname[:-4] + '-annot.fif'
    if not op.isfile(annot_fname):
        if np.isfinite(lims[:3]).any() or np.isfinite(lims[5]):
            print(prefix.join(['', 'Annotating raw segments with:\n',
                               u'  rotation_limit    = %s °/s\n' % lims[0],
                               u'  translation_limit = %s m/s\n' % lims[1],
                               u'  coil_dist_limit   = %s m\n' % lims[2],
                               u'  t_step, t_window  = %s, %s sec\n'
                               % (lims[3], lims[4]),
                               u'  3-good limit      = %s sec' % (lims[5],)]))
        annot = annotate_head_pos(
            raw, head_pos, rotation_limit=lims[0], translation_limit=lims[1],
            fit_t=fit_data['fit_t'], counts=fit_data['counts'],
            prefix='  ' + prefix, coil_bad_count_duration_limit=lims[5])
        if annot is not None:
            annot.save(annot_fname)
    try:
        annot = read_annotations(annot_fname)
    except IOError:  # no annotations requested
        annot = None
    return head_pos, annot, fit_data


def info_sss_basis(info, origin='auto', int_order=8, ext_order=3,
                   coord_frame='head', regularize='in', ignore_ref=True):
    """Compute the SSS basis for a given measurement info structure

    Parameters
    ----------
    info : instance of io.Info
        The measurement info.
    origin : array-like, shape (3,) | str
        Origin of internal and external multipolar moment space in meters.
        The default is ``'auto'``, which means a head-digitization-based
        origin fit when ``coord_frame='head'``, and ``(0., 0., 0.)`` when
        ``coord_frame='meg'``.
    int_order : int
        Order of internal component of spherical expansion.
    ext_order : int
        Order of external component of spherical expansion.
    coord_frame : str
        The coordinate frame that the ``origin`` is specified in, either
        ``'meg'`` or ``'head'``. For empty-room recordings that do not have
        a head<->meg transform ``info['dev_head_t']``, the MEG coordinate
        frame should be used.
    destination : str | array-like, shape (3,) | None
        The destination location for the head. Can be ``None``, which
        will not change the head position, or a string path to a FIF file
        containing a MEG device<->head transformation, or a 3-element array
        giving the coordinates to translate to (with no rotations).
        For example, ``destination=(0, 0, 0.04)`` would translate the bases
        as ``--trans default`` would in MaxFilter™ (i.e., to the default
        head location).
    regularize : str | None
        Basis regularization type, must be "in", "svd" or None.
        "in" is the same algorithm as the "-regularize in" option in
        MaxFilter™. "svd" (new in v0.13) uses SVD-based regularization by
        cutting off singular values of the basis matrix below the minimum
        detectability threshold of an ideal head position (usually near
        the device origin).
    ignore_ref : bool
        If True, do not include reference channels in compensation. This
        option should be True for KIT files, since Maxwell filtering
        with reference channels is not currently supported.
    """
    if coord_frame not in ('head', 'meg'):
        raise ValueError('coord_frame must be either "head" or "meg", not "%s"'
                         % coord_frame)
    origin = _check_origin(origin, info, 'head')
    regularize = _check_regularize(regularize, ('in', 'svd'))
    meg_picks, mag_picks, grad_picks, good_picks, coil_scale, mag_or_fine = \
        _get_mf_picks(info, int_order, ext_order, ignore_ref)
    info_good = pick_info(info, good_picks, copy=True)
    all_coils = _prep_mf_coils(info_good, ignore_ref=ignore_ref)
    # remove MEG bads in "to" info
    decomp_coil_scale = coil_scale[good_picks]
    exp = dict(int_order=int_order, ext_order=ext_order, head_frame=True,
               origin=origin)
    # prepare regularization techniques
    if _prep_regularize is None:
        raise RuntimeError('mne-python needs to be on the experimental SVD '
                           'branch to use this function')
    _prep_regularize(regularize, all_coils, None, exp, ignore_ref,
                     coil_scale, grad_picks, mag_picks, mag_or_fine)
    # noinspection PyPep8Naming
    S = _trans_sss_basis(exp, all_coils, info['dev_head_t'],
                         coil_scale=decomp_coil_scale)
    if regularize is not None:
        # noinspection PyPep8Naming
        S = _regularize(regularize, exp, S, mag_or_fine, t=0.)[0]
    S /= np.linalg.norm(S, axis=0)
    return S


def clean_brain(brain_img):
    """Remove borders of a brain image and make transparent."""
    bg = (brain_img == brain_img[0, 0]).all(-1)
    brain_img = brain_img[(~bg).any(axis=-1)]
    brain_img = brain_img[:, (~bg).any(axis=0)]
    alpha = 255 * np.ones(brain_img.shape[:-1], np.uint8)
    x, y = np.where((brain_img == 255).all(-1))
    alpha[x, y] = 0
    return np.concatenate((brain_img, alpha[..., np.newaxis]), -1)


def plot_colorbar(pos_lims, ticks=None, ticklabels=None, figsize=(1, 2),
                  labelsize='small', ticklabelsize='x-small', ax=None,
                  label='', tickrotation=0., orientation='vertical',
                  end_labels=None):
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    with plt.rc_context({'axes.labelsize': labelsize,
                         'xtick.labelsize': ticklabelsize,
                         'ytick.labelsize': ticklabelsize}):
        cmap = mne.viz.utils.mne_analyze_colormap(
            limits=pos_lims, format='matplotlib')
        adjust = (ax is None)
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            fig = ax.figure
        norm = Normalize(vmin=-pos_lims[2], vmax=pos_lims[2])
        if ticks is None:
            ticks = [-pos_lims[2], -pos_lims[1], -pos_lims[0], 0.,
                     pos_lims[0], pos_lims[1], pos_lims[2]]
        if ticklabels is None:
            ticklabels = ticks
        assert len(ticks) == len(ticklabels)
        cbar = ColorbarBase(ax, cmap, norm=norm, ticks=ticks, label=label,
                            orientation=orientation)
        for key in ('left', 'top',
                    'bottom' if orientation == 'vertical' else 'right'):
            ax.spines[key].set_visible(False)
        cbar.set_ticklabels(ticklabels)
        cbar.patch.set(facecolor='0.5', edgecolor='0.5')
        if orientation == 'horizontal':
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=tickrotation)
        else:
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=tickrotation)
        cbar.outline.set_visible(False)
        lims = np.array(list(ax.get_xlim()) + list(ax.get_ylim()))
        if end_labels is not None:
            if orientation == 'horizontal':
                delta = np.diff(lims[:2]) * np.array([-0.05, 0.05])
                xs = np.array(lims[:2]) + delta
                has = ['right', 'left']
                ys = [lims[2:].mean()] * 2
                vas = ['center', 'center']
            else:
                xs = [lims[:2].mean()] * 2
                has = ['center'] * 2
                delta = np.diff(lims[2:]) * np.array([-0.05, 0.05])
                ys = lims[2:] + delta
                vas = ['top', 'bottom']
            for x, y, l, ha, va in zip(xs, ys, end_labels, has, vas):
                ax.text(x, y, l, ha=ha, va=va, fontsize=ticklabelsize)
        if adjust:
            fig.subplots_adjust(0.01, 0.05, 0.2, 0.95)
    return fig


def plot_reconstruction(evoked, origin=(0., 0., 0.04)):
    """Plot the reconstructed data for Evoked

    Currently only works for MEG data.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data.
    origin : array-like, shape (3,)
        The head origin to use.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    from mne.forward._field_interpolation import _map_meg_channels
    import matplotlib.pyplot as plt
    evoked = evoked.copy().pick_types(meg=True, exclude='bads')
    info_to = deepcopy(evoked.info)
    info_to['projs'] = []
    op = _map_meg_channels(
        evoked.info, info_to, mode='accurate', origin=(0., 0., 0.04))
    fig, axs = plt.subplots(3, 2, squeeze=False)
    titles = dict(grad='Gradiometers (fT/cm)', mag='Magnetometers (fT)')
    for mi, meg in enumerate(('grad', 'mag')):
        picks = pick_types(evoked.info, meg=meg)
        kwargs = dict(ylim=dict(grad=[-250, 250], mag=[-600, 600]),
                      spatial_colors=True, picks=picks)
        evoked.plot(axes=axs[0, mi], proj=False,
                    titles=dict(grad='Proj off', mag=''), **kwargs)
        evoked_remap = evoked.copy().apply_proj()
        evoked_remap.info['projs'] = []
        evoked_remap.plot(axes=axs[1, mi],
                          titles=dict(grad='Proj on', mag=''), **kwargs)
        evoked_remap.data = np.dot(op, evoked_remap.data)
        evoked_remap.plot(axes=axs[2, mi],
                          titles=dict(grad='Recon', mag=''), **kwargs)
        axs[0, mi].set_title(titles[meg])
        for ii in range(3):
            if ii in (0, 1):
                axs[ii, mi].set_xlabel('')
            if ii in (1, 2):
                axs[ii, mi].set_title('')
    for ii in range(3):
        axs[ii, 1].set_ylabel('')
    axs[0, 0].set_ylabel('Original')
    axs[1, 0].set_ylabel('Projection')
    axs[2, 0].set_ylabel('Reconstruction')
    fig.tight_layout()
    return fig


def plot_chpi_snr_raw(raw, win_length, n_harmonics=None, show=True):
    """Compute and plot cHPI SNR from raw data

    Parameters
    ----------
    win_length : float
        Length of window to use for SNR estimates (seconds). A longer window
        will naturally include more low frequency power, resulting in lower
        SNR.
    n_harmonics : int or None
        Number of line frequency harmonics to include in the model. If None,
        use all harmonics up to the MEG analog lowpass corner.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        cHPI SNR as function of time, residual variance.

    Notes
    -----
    A general linear model including cHPI and line frequencies is fit into
    each data window. The cHPI power obtained from the model is then divided
    by the residual variance (variance of signal unexplained by the model) to
    obtain the SNR.

    The SNR may decrease either due to decrease of cHPI amplitudes (e.g.
    head moving away from the helmet), or due to increase in the residual
    variance. In case of broadband interference that overlaps with the cHPI
    frequencies, the resulting decreased SNR accurately reflects the true
    situation. However, increased narrowband interference outside the cHPI
    and line frequencies would also cause an increase in the residual variance,
    even though it wouldn't necessarily affect estimation of the cHPI
    amplitudes. Thus, this method is intended for a rough overview of cHPI
    signal quality. A more accurate picture of cHPI quality (at an increased
    computational cost) can be obtained by examining the goodness-of-fit of
    the cHPI coil fits.
    """
    import matplotlib.pyplot as plt

    # plotting parameters
    legend_fontsize = 10
    title_fontsize = 10
    tick_fontsize = 10
    label_fontsize = 10

    # get some info from fiff
    sfreq = raw.info['sfreq']
    linefreq = raw.info['line_freq']
    if n_harmonics is not None:
        linefreqs = (np.arange(n_harmonics + 1) + 1) * linefreq
    else:
        linefreqs = np.arange(linefreq, raw.info['lowpass'], linefreq)
    buflen = int(win_length * sfreq)
    if buflen <= 0:
        raise ValueError('Window length should be >0')
    (cfreqs, _, _, _, _) = _get_hpi_info(raw.info)
    print('Nominal cHPI frequencies: %s Hz' % cfreqs)
    print('Sampling frequency: %s Hz' % sfreq)
    print('Using line freqs: %s Hz' % linefreqs)
    print('Using buffers of %s samples = %s seconds\n'
          % (buflen, buflen/sfreq))

    pick_meg = pick_types(raw.info, meg=True, exclude=[])
    pick_mag = pick_types(raw.info, meg='mag', exclude=[])
    pick_grad = pick_types(raw.info, meg='grad', exclude=[])
    nchan = len(pick_meg)
    # grad and mag indices into an array that already has meg channels only
    pick_mag_ = np.in1d(pick_meg, pick_mag).nonzero()[0]
    pick_grad_ = np.in1d(pick_meg, pick_grad).nonzero()[0]

    # create general linear model for the data
    t = np.arange(buflen) / float(sfreq)
    model = np.empty((len(t), 2+2*(len(linefreqs)+len(cfreqs))))
    model[:, 0] = t
    model[:, 1] = np.ones(t.shape)
    # add sine and cosine term for each freq
    allfreqs = np.concatenate([linefreqs, cfreqs])
    model[:, 2::2] = np.cos(2 * np.pi * t[:, np.newaxis] * allfreqs)
    model[:, 3::2] = np.sin(2 * np.pi * t[:, np.newaxis] * allfreqs)
    inv_model = linalg.pinv(model)

    # drop last buffer to avoid overrun
    bufs = np.arange(0, raw.n_times, buflen)[:-1]
    tvec = bufs/sfreq
    snr_avg_grad = np.zeros([len(cfreqs), len(bufs)])
    hpi_pow_grad = np.zeros([len(cfreqs), len(bufs)])
    snr_avg_mag = np.zeros([len(cfreqs), len(bufs)])
    resid_vars = np.zeros([nchan, len(bufs)])
    for ind, buf0 in enumerate(bufs):
        print('Buffer %s/%s' % (ind+1, len(bufs)))
        megbuf = raw[pick_meg, buf0:buf0+buflen][0].T
        coeffs = np.dot(inv_model, megbuf)
        coeffs_hpi = coeffs[2+2*len(linefreqs):]
        resid_vars[:, ind] = np.var(megbuf-np.dot(model, coeffs), 0)
        # get total power by combining sine and cosine terms
        # sinusoidal of amplitude A has power of A**2/2
        hpi_pow = (coeffs_hpi[0::2, :]**2 + coeffs_hpi[1::2, :]**2)/2
        hpi_pow_grad[:, ind] = hpi_pow[:, pick_grad_].mean(1)
        # divide average HPI power by average variance
        snr_avg_grad[:, ind] = hpi_pow_grad[:, ind] / \
            resid_vars[pick_grad_, ind].mean()
        snr_avg_mag[:, ind] = hpi_pow[:, pick_mag_].mean(1) / \
            resid_vars[pick_mag_, ind].mean()

    cfreqs_legend = ['%s Hz' % fre for fre in cfreqs]
    fig, axs = plt.subplots(4, 1, sharex=True)

    # SNR plots for gradiometers and magnetometers
    ax = axs[0]
    lines1 = ax.plot(tvec, 10*np.log10(snr_avg_grad.T))
    lines1_med = ax.plot(tvec, 10*np.log10(np.median(snr_avg_grad, axis=0)),
                         lw=2, ls=':', color='k')
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='SNR (dB)')
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Mean cHPI power / mean residual variance, gradiometers',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax = axs[1]
    lines2 = ax.plot(tvec, 10*np.log10(snr_avg_mag.T))
    lines2_med = ax.plot(tvec, 10 * np.log10(np.median(snr_avg_mag, axis=0)),
                         lw=2, ls=':', color='k')
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='SNR (dB)')
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Mean cHPI power / mean residual variance, magnetometers',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax = axs[2]
    lines3 = ax.plot(tvec, hpi_pow_grad.T)
    lines3_med = ax.plot(tvec, np.median(hpi_pow_grad, axis=0),
                         lw=2, ls=':', color='k')
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='Power (T/m)$^2$')
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Mean cHPI power, gradiometers',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # residual (unexplained) variance as function of time
    ax = axs[3]
    cls = plt.get_cmap('plasma')(np.linspace(0., 0.7, len(pick_meg)))
    ax.set_prop_cycle(color=cls)
    ax.semilogy(tvec, resid_vars[pick_grad_, :].T, alpha=.4)
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='Var. (T/m)$^2$', xlabel='Time (s)')
    ax.xaxis.label.set_fontsize(label_fontsize)
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Residual (unexplained) variance, all gradiometer channels',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    tight_layout(pad=.5, w_pad=.1, h_pad=.2)  # from mne.viz
    # tight_layout will screw these up
    ax = axs[0]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # order curve legends according to mean of data
    sind = np.argsort(snr_avg_grad.mean(axis=1))[::-1]
    handles = [lines1[i] for i in sind]
    handles.append(lines1_med[0])
    labels = [cfreqs_legend[i] for i in sind]
    labels.append('Median')
    ax.legend(handles, labels,
              prop={'size': legend_fontsize}, bbox_to_anchor=(1.02, 0.5, ),
              loc='center left', borderpad=1)
    ax = axs[1]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    sind = np.argsort(snr_avg_mag.mean(axis=1))[::-1]
    handles = [lines2[i] for i in sind]
    handles.append(lines2_med[0])
    labels = [cfreqs_legend[i] for i in sind]
    labels.append('Median')
    ax.legend(handles, labels,
              prop={'size': legend_fontsize}, bbox_to_anchor=(1.02, 0.5, ),
              loc='center left', borderpad=1)
    ax = axs[2]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    sind = np.argsort(hpi_pow_grad.mean(axis=1))[::-1]
    handles = [lines3[i] for i in sind]
    handles.append(lines3_med[0])
    labels = [cfreqs_legend[i] for i in sind]
    labels.append('Median')
    ax.legend(handles, labels,
              prop={'size': legend_fontsize}, bbox_to_anchor=(1.02, 0.5, ),
              loc='center left', borderpad=1)
    ax = axs[3]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.show(show)

    return fig


@verbose
def compute_good_coils(raw, t_step=0.01, t_window=0.2, dist_limit=0.005,
                       prefix='', verbose=None):
    """Comute time-varying coil distances."""
    from scipy.spatial.distance import cdist
    hpi_dig_head_rrs = _get_hpi_initial_fit(raw.info, verbose=False)
    n_window = (int(round(t_window * raw.info['sfreq'])) // 2) * 2 + 1
    del t_window
    hpi = _setup_hpi_struct(raw.info, n_window, verbose=False)
    hpi_coil_dists = cdist(hpi_dig_head_rrs, hpi_dig_head_rrs)
    n_step = int(round(t_step * raw.info['sfreq']))
    del t_step
    starts = np.arange(0, len(raw.times) - n_window // 2, n_step)
    counts = np.empty(len(starts), int)
    head_dev_t = mne.transforms.invert_transform(
        raw.info['dev_head_t'])['trans']
    coil_dev_rrs = mne.transforms.apply_trans(head_dev_t, hpi_dig_head_rrs)
    last_fit = None
    last = -10.
    logger.info('%sComputing %d coil fits in %0.1f ms steps over %0.1f sec'
                % (prefix, len(starts), (n_step / raw.info['sfreq']) * 1000,
                   raw.times[-1]))
    for ii, start in enumerate(starts):
        time_sl = slice(max(start - n_window // 2, 0), start + n_window // 2)
        t = start / raw.info['sfreq']
        if t - last >= 10. - 1e-7:
            logger.info('%s    Fitting %0.1f - %0.1f sec'
                        % (prefix, t, min(t + 10., raw.times[-1])))
            last = t
        # Ignore warnings about segments with not enough coils on
        sin_fit = _fit_cHPI_amplitudes(raw, time_sl, hpi, t, verbose=False)
        # skip this window if it bad.
        if sin_fit is None:
            counts[ii] = 0
            continue

        # check if data has sufficiently changed
        if last_fit is not None:  # first iteration
            # The sign of our fits is arbitrary
            flips = np.sign((sin_fit * last_fit).sum(-1, keepdims=True))
            sin_fit *= flips
            corr = np.corrcoef(sin_fit.ravel(), last_fit.ravel())[0, 1]
            # check to see if we need to continue
            if corr * corr > 0.98:
                # don't need to refit data
                counts[ii] = counts[ii - 1]
                continue

        last_fit = sin_fit.copy()

        kwargs = dict()
        if 'too_close' in get_args(_fit_magnetic_dipole):
            kwargs['too_close'] = 'warning'

        outs = [_fit_magnetic_dipole(f, pos, hpi['coils'], hpi['scale'],
                                     hpi['method'], **kwargs)
                for f, pos in zip(sin_fit, coil_dev_rrs)]

        coil_dev_rrs = np.array([o[0] for o in outs])
        these_dists = cdist(coil_dev_rrs, coil_dev_rrs)
        these_dists = np.abs(hpi_coil_dists - these_dists)
        # there is probably a better algorithm for finding the bad ones...
        use_mask = np.ones(hpi['n_freqs'], bool)
        good = False
        while not good:
            d = these_dists[use_mask][:, use_mask]
            d_bad = (d > dist_limit)
            good = not d_bad.any()
            if not good:
                if use_mask.sum() == 2:
                    use_mask[:] = False
                    break  # failure
                # exclude next worst point
                badness = (d * d_bad).sum(axis=0)
                exclude_coils = np.where(use_mask)[0][np.argmax(badness)]
                use_mask[exclude_coils] = False
        counts[ii] = use_mask.sum()
    t = (starts + n_window // 2) / raw.info['sfreq']
    return t, counts, len(hpi_dig_head_rrs)


@verbose
def plot_good_coils(raw, t_step=1., t_window=0.2, dist_limit=0.005,
                    show=True, verbose=None):
    """Plot the good coil count as a function of time."""
    import matplotlib.pyplot as plt
    if isinstance(raw, dict):  # fit_data calculated and stored to disk
        t = raw['fit_t']
        counts = raw['counts']
        n_coils = raw['n_coils']
    else:
        t, counts, n_coils = compute_good_coils(raw, t_step, t_window,
                                                dist_limit)
    del t_step, t_window, dist_limit
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.step(t, counts, zorder=4, color='k', clip_on=False)
    ax.set(xlim=t[[0, -1]], ylim=[0, n_coils], xlabel='Time (sec)',
           ylabel='Good coils')
    ax.set(yticks=np.arange(n_coils + 1))
    for comp, n, color in ((np.greater_equal, 5, '#2ca02c'),
                           (np.equal, 4, '#98df8a'),
                           (np.equal, 3, (1, 1, 0)),
                           (np.less_equal, 2, (1, 0, 0))):
        mask = comp(counts, n)
        mask[:-1] |= comp(counts[1:], n)
        ax.fill_between(t, 0, n_coils, where=mask,
                        color=color, edgecolor='none', linewidth=0, zorder=1)
    ax.grid(True)
    fig.tight_layout()
    mne.viz.utils.plt_show(show)
    return fig


def compute_auc(dip, tmin=-np.inf, tmax=np.inf):
    """Compute the AUC values for a DipoleFixed object."""
    if not isinstance(dip, DipoleFixed):
        raise TypeError('dip must be a DipoleFixed, got "%s"' % (type(dip),))
    pick = pick_types(dip.info, meg=False, dipole=True)
    if len(pick) != 1:
        raise RuntimeError('Could not find dipole data')
    time_mask = _time_mask(dip.times, tmin, tmax, dip.info['sfreq'])
    data = dip.data[pick[0], time_mask]
    return np.sum(np.abs(data)) * len(data) * (1. / dip.info['sfreq'])


def _spherical_conductor(info, subject, pos):
    """Helper to make spherical conductor model."""
    bem = make_sphere_model(info=info, r0='auto',
                            head_radius='auto', verbose=False)
    src = setup_volume_source_space(subject=subject, sphere=bem,
                                    pos=pos, mindist=1.)
    return bem, src, None


def annotate_head_pos(raw, head_pos, rotation_limit=45, translation_limit=0.1,
                      fit_t=None, counts=None, prefix='  ',
                      coil_bad_count_duration_limit=0.1):
    u"""Annotate a raw instance based on coil counts and head positions.

    Parameters
    ----------
    raw : instance of Raw
        The raw instance.
    head_pos : ndarray | None
        The head positions. Can be None if movement compensation is off
        to short-circuit the function.
    rotation_limit : float
        The rotational velocity limit in °/s.
        Can be infinite to skip rotation checks.
    translation_limit : float
        The translational velocity limit in m/s.
        Can be infinite to skip translation checks.
    fit_t : ndarray
        Fit times.
    counts : ndarray
        Coil counts.
    prefix : str
        The prefix for printing.
    coil_bad_count_duration_limit : float
        The lower limit for bad coil counts to remove segments of data.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """
    # XXX: Add `sphere_dist_limit` to ensure no sensor collisions at some
    # point
    do_rotation = np.isfinite(rotation_limit) and head_pos is not None
    do_translation = np.isfinite(translation_limit) and head_pos is not None
    do_coils = fit_t is not None and counts is not None
    if not (do_rotation or do_translation or do_coils):
        return None
    head_pos_t = head_pos[:, 0]
    dt = np.diff(head_pos_t)

    annot = mne.Annotations([], [], [])

    # Annotate based on bad coil distances
    if do_coils:
        if np.isfinite(coil_bad_count_duration_limit):
            changes = np.diff((counts < 3).astype(int))
            bad_onsets = fit_t[np.where(changes == 1)[0]]
            bad_offsets = fit_t[np.where(changes == -1)[0]]
            # Deal with it starting out bad
            if counts[0] < 3:
                bad_onsets = np.concatenate([[0.], bad_onsets])
            if counts[-1] < 3:
                bad_offsets = np.concatenate([bad_offsets, [raw.times[-1]]])
            assert len(bad_onsets) == len(bad_offsets)
            assert (bad_onsets[1:] > bad_offsets[:-1]).all()
            count = 0
            dur = 0.
            for onset, offset in zip(bad_onsets, bad_offsets):
                if offset - onset > coil_bad_count_duration_limit - 1e-6:
                    annot.append(onset, offset - onset, 'BAD_HPI_COUNT')
                    dur += offset - onset
                    count += 1
            print('%sOmitting %5.1f%% (%3d segments) '
                  'due to < 3 good coils for over %s sec'
                  % (prefix, 100 * dur / raw.times[-1], count,
                     coil_bad_count_duration_limit))

    # Annotate based on rotational velocity
    if do_rotation:
        assert rotation_limit > 0
        # Rotational velocity (radians / sec)
        r = mne.transforms._angle_between_quats(head_pos[:-1, 1:4],
                                                head_pos[1:, 1:4])
        r /= dt
        bad_idx = np.where(r >= np.deg2rad(rotation_limit))[0]
        bad_pct = 100 * dt[bad_idx].sum() / (head_pos[-1, 0] - head_pos[0, 0])
        print(u'%sOmitting %5.1f%% (%3d segments) due to bad rotational '
              'velocity (>=%5.1f deg/s), with max %0.2f deg/s'
              % (prefix, bad_pct, len(bad_idx), rotation_limit,
                 np.rad2deg(r.max())))
        for idx in bad_idx:
            annot.append(head_pos_t[idx], dt[idx], 'BAD_RV')

    # Annotate based on translational velocity
    if do_translation:
        assert translation_limit > 0
        v = np.linalg.norm(np.diff(head_pos[:, 4:7], axis=0), axis=-1)
        v /= dt
        bad_idx = np.where(v >= translation_limit)[0]
        bad_pct = 100 * dt[bad_idx].sum() / (head_pos[-1, 0] - head_pos[0, 0])
        print(u'%sOmitting %5.1f%% (%3d segments) due to translational '
              u'velocity (>=%5.1f m/s), with max %0.4f m/s'
              % (prefix, bad_pct, len(bad_idx), translation_limit, v.max()))
        for idx in bad_idx:
            annot.append(head_pos_t[idx], dt[idx], 'BAD_TV')

    # Annotate on distance from the sensors
    return annot


@contextmanager
def mlab_offscreen(offscreen=True):
    from mayavi import mlab
    old_offscreen = mlab.options.offscreen
    mlab.options.offscreen = offscreen
    yield
    mlab.options.offscreen = old_offscreen


def discretize_cmap(colormap, lims, transparent=True):
    """Discretize a colormap."""
    lims = np.array(lims, int)
    assert lims.shape == (2,)
    from matplotlib import colors, pyplot as plt
    n_pts = lims[1] - lims[0] + 1
    assert n_pts > 0
    if n_pts == 1:
        vals = np.ones(256)
    else:
        vals = np.round(np.linspace(-0.5, n_pts - 0.5, 256)) / (n_pts - 1)
    colormap = plt.get_cmap(colormap)(vals)
    if transparent:
        colormap[:, 3] = np.clip((vals + 0.5 / n_pts) * 2, 0, 1)
    colormap[0, 3] = 0.
    colormap = colors.ListedColormap(colormap)
    use_lims = [lims[0] - 0.5, (lims[0] + lims[1]) / 2., lims[1] + 0.5]
    return colormap, use_lims
