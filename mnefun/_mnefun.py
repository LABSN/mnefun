# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import print_function

import os
import os.path as op
import inspect
import warnings
from shutil import move, copy2
import subprocess
import glob
from collections import Counter
from time import time

import numpy as np
from scipy import io as spio
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from mne import (compute_proj_raw, make_fixed_length_events, Epochs,
                 find_events, read_events, write_events, concatenate_events,
                 read_cov, compute_covariance, write_cov, read_forward_solution,
                 write_proj, read_proj, setup_source_space,
                 make_forward_solution, get_config, write_evokeds,
                 make_sphere_model, setup_volume_source_space,
                 read_bem_solution)
try:
    from mne import compute_raw_covariance  # up-to-date mne-python
except ImportError:  # oldmne-python
    from mne import compute_raw_data_covariance as compute_raw_covariance
from mne.preprocessing.ssp import compute_proj_ecg, compute_proj_eog
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.preprocessing.maxwell import maxwell_filter
from mne.minimum_norm import make_inverse_operator
from mne.label import read_label
from mne.epochs import combine_event_ids
try:
    from mne.chpi import _quat_to_rot, _rot_to_quat
except ImportError:
    from mne.io.chpi import _quat_to_rot, _rot_to_quat
from mne.io import Raw, concatenate_raws, read_info, write_info
from mne.io.pick import pick_types_forward, pick_types
from mne.io.meas_info import _empty_info
from mne.cov import regularize
from mne.minimum_norm import write_inverse_operator
from mne.viz import plot_drop_log
from mne.utils import run_subprocess
from mne.report import Report
from mne.io.constants import FIFF

from ._paths import (get_raw_fnames, get_event_fnames, get_report_fnames,
                     get_epochs_evokeds_fnames, safe_inserter, _regex_convert)
from ._status import print_proc_status
from ._reorder import fix_eeg_channels
from ._scoring import default_score

# python2/3 conversions
try:
    string_types = basestring  # noqa, analysis:ignore
except Exception:
    string_types = str

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
    movecomp : str | None
        Movement compensation to use. Can be 'inter' or None.
    sss_type : str
        signal space separation method. Must be either 'maxfilter' or 'python'
    int_order : int
        Order of internal component of spherical expansion.
    ext_order : int
        Order of external component of spherical expansion.
    tsss_dur : float | None
        Apply spatiotemporal SSS with specified buffer duration
        (in seconds). Elekta's default is 10.0 seconds in MaxFilter™ v2.2.
        Spatiotemporal SSS acts as implicitly as a high-pass filter where the
        cut-off frequency is 1/st_dur Hz. For this (and other) reasons, longer
        buffers are generally better as long as your system can handle the
        higher memory usage. To ensure that each window is processed
        identically, choose a buffer length that divides evenly into your data.
        Any data at the trailing edge that doesn't fit evenly into a whole
        buffer window will be lumped into the previous buffer.
    st_correlation : float
        Correlation limit between inner and outer subspaces used to reject
        ovwrlapping intersecting inner/outer signals during spatiotemporal SSS.
    trans_to : str | array-like, shape (3,) | None
        The destination location for the head. Can be ``None``, which
        will not change the head position, or a string path to a FIF file
        containing a MEG device<->head transformation, or a 3-element array
        giving the coordinates to translate to (with no rotations).
        For example, ``destination=(0, 0, 0.04)`` would translate the bases
        as ``--trans default`` would in MaxFilter™ (i.e., to the default
        head location).
    origin : array-like, shape (3,) | str
        Origin of internal and external multipolar moment space in meters.
        The default is ``'auto'``, which means ``(0., 0., 0.)`` for
        ``coord_frame='meg'``, and a head-digitization-based origin fit
        for ``coord_frame='head'``.

    Returns
    -------
    params : instance of Params
        The parameters to use.

    See also
    --------
    do_processing
    run_sss_localy
    mne.maxwell_filter

    Notes
    -----
    - Params has additional properties. Use ``dir(params)`` to see
    all the possible options.
    - For Maxwell filtering with mne.maxwell_filter the following default
    parameters:
        * tSSS correlation = 0.98
        * Order of internal component of spherical expansion = 3
        * Order of external component of spherical expansion = 8
        * tSSS buffer length = 60 seconds
        * Spherical expansion coordinate frame = head
        * Coordinate frame origin =  head-digitization-based origin fit

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
                 lp_trans=0.5, hp_trans=0.5, movecomp='inter',
                 sss_type='maxfilter', int_order=8, ext_order=3,
                 st_correlation=0.98, sss_origin='head'):
        self.reject = dict(eog=np.inf, grad=1500e-13, mag=5000e-15, eeg=150e-6)
        self.flat = dict(eog=-1, grad=1e-13, mag=1e-15, eeg=1e-6)
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
        self.n_jobs_fir = 'cuda'  # Jobs when using method='fft'
        self.n_jobs_resample = n_jobs_resample
        self.filter_length = filter_length
        self.cont_lp = 5
        self.lp_cut = lp_cut
        self.hp_cut = hp_cut
        self.lp_trans = lp_trans
        self.hp_trans = hp_trans
        self.disp_files = True
        self.plot_drop_logs = True  # plot drop logs after do_preprocessing_...
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
        # Maxfilter params
        self.mf_args = ''
        self.tsss_dur = 60.
        # boolean for whether data set(s) have an individual mri
        self.on_process = None
        # Use more than EXTRA points to fit headshape
        self.dig_with_eeg = False
        # Function to pick a subset of events to use to make a covariance
        self.pick_events_cov = lambda x: x
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
        self.trans_to = 'median'  # where to transform head positions to
        self.sss_format = 'float'  # output type for MaxFilter
        self.subject_run_indices = None
        self.movecomp = movecomp
        assert self.movecomp in ('inter', None)
        # Maxwell filtering parameters
        self.sss_type = sss_type
        assert self.sss_type in ('maxfilter', 'python')
        self.int_order = int_order
        self.ext_order = ext_order
        self.st_correlation = st_correlation
        self.sss_origin = sss_origin
        self.freeze()

    @property
    def pca_extra(self):
        return '_allclean_fil%d' % self.lp_cut

    @property
    def pca_fif_tag(self):
        return self.pca_extra + self.sss_fif_tag


def _get_baseline(p):
    """Helper to extract baseline from params"""
    if p.baseline == 'individual':
        baseline = (p.bmin, p.bmax)
    else:
        baseline = p.baseline
    return baseline


def do_processing(p, fetch_raw=False, do_score=False, push_raw=False,
                  do_sss=False, fetch_sss=False, do_ch_fix=False,
                  gen_ssp=False, apply_ssp=False, plot_psd=False,
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
    plot_psd : bool
        Plot continuous raw data power spectra
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
    bools = [fetch_raw,
             do_score,
             push_raw,
             do_sss,
             fetch_sss,
             do_ch_fix,
             gen_ssp,
             apply_ssp,
             plot_psd,
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
             'Plotting raw data power',
             'Doing epoch EQ/DQ',
             'Generating covariances',
             'Generating forward models',
             'Generating inverse solutions',
             'Generating HTML Reports',
             'Status',
             ]
    score_fun = p.score if p.score is not None else default_score
    if len(inspect.getargspec(score_fun).args) == 2:
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
             plot_raw_psd,
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
        dates = [tuple([int(dd) for dd in d])
                 for d in np.array(p.dates)[sinds]]

    decim = p.decim
    if not isinstance(decim, (list, tuple)):
        decim = [decim] * len(p.subjects)
    assert len(decim) == n_subj_orig
    decim = np.array(decim)
    assert decim.dtype == np.int64
    assert decim.ndim == 1
    assert decim.size == len(p.subjects)
    decim = decim[sinds]

    run_indices = p.subject_run_indices
    if run_indices is None:
        run_indices = [None] * len(p.subjects)
    assert len(run_indices) == len(p.subjects)
    run_indices = [r for ri, r in enumerate(run_indices) if ri in sinds]
    assert all(r is None or np.in1d(r, np.arange(len(p.run_names))).all()
               for r in run_indices)

    if p.sss_type is 'python' and fetch_sss:
        raise RuntimeError(' You are running SSS pre-processing locally '
                           ' and attempting to pull SSS files from remote workstation. '
                           ' Set fetch_sss parameter to False and try again!')
    # Actually do the work

    outs = [None] * len(bools)
    for ii, (b, text, func) in enumerate(zip(bools, texts, funcs)):
        if b:
            t0 = time()
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
            print('  (' + timestring(time() - t0) + ')')
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
        finder_stem = 'find %s ' % p.acq_dir
        # build remote raw file finder
        fnames = get_raw_fnames(p, subj, 'raw', True, False, run_indices[si])
        assert len(fnames) > 0
        finder = (finder_stem +
                  ' -o '.join(['-type f -regex ' + _regex_convert(f)
                               for f in fnames]))
        stdout_ = run_subprocess(['ssh', '-p', str(p.acq_port),
                                  p.acq_ssh, finder])[0]
        remote_fnames = [x.strip() for x in stdout_.splitlines()]
        assert all(fname.startswith(p.acq_dir) for fname in remote_fnames)
        remote_fnames = [fname[len(p.acq_dir) + 1:] for fname in remote_fnames]
        want = set(op.basename(fname) for fname in fnames)
        got = set(op.basename(fname) for fname in remote_fnames)
        if want != got.intersection(want):
            raise RuntimeError('Could not find all files.\nWanted: %s\nGot: %s'
                               % (want, got.intersection(want)))
        if len(remote_fnames) != len(fnames):
            warnings.warn('Found more files than expected on remote server.\n'
                          'Likely split files were found. Please confirm '
                          'results.')
        print('  Pulling %s files for %s...' % (len(remote_fnames), subj))
        cmd = ['rsync', '-ave', 'ssh -p %s' % p.sws_port,
               '--prune-empty-dirs', '--partial',
               '--include', '*/']
        for fname in remote_fnames:
            cmd += ['--include', op.basename(fname)]
        remote_loc = '%s:%s' % (p.acq_ssh, op.join(p.acq_dir, ''))
        cmd += ['--exclude', '*', remote_loc, op.join(raw_dir, '')]
        run_subprocess(cmd)
        # move files to root raw_dir
        for fname in remote_fnames:
            move(op.join(raw_dir, fname), op.join(raw_dir, op.basename(fname)))
        # prune the extra directories we made
        for fname in remote_fnames:
            next_ = op.split(fname)[0]
            while len(next_) > 0:
                if op.isdir(op.join(raw_dir, next_)):
                    os.rmdir(op.join(raw_dir, next_))  # safe; goes if empty
                next_ = op.split(next_)[0]


def calc_median_hp(p, subj, out_file, ridx):
    """Calculate median head position"""
    print('    Estimating median head position for %s... ' % subj)
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
        assert_allclose(np.linalg.det(m), 1., atol=1e-5) #for the determinant
        qs.append(_rot_to_quat(m))
    assert info is not None
    if len(raw_files) == 1:  # only one head position
        dev_head_t = info['dev_head_t']
    else:
        t = np.median(np.array(ts), axis=0)
        rot = np.median(_quat_to_rot(np.array(qs)), axis=0)
        trans = np.r_[np.c_[rot, t[:, np.newaxis]],
                      np.array([0, 0, 0, 1], t.dtype)[np.newaxis, :]]
        dev_head_t = {'to': 4, 'from': 1, 'trans': trans}
    info = _empty_info(info['sfreq'])
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
                                                  dig_kinds=dig_kinds)[1]
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
    run_subprocess(cmd, cwd=p.work_dir)


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
    if p.sss_type is 'python':
        print(' Applying SSS locally using mne-python')
        run_sss_localy(p, subjects, run_indices)
    else:
        for si, subj in enumerate(subjects):
            files = get_raw_fnames(p, subj, 'raw', False, True, run_indices[si])
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
            run_subprocess(cmd, stdout=None, stderr=None)
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
    run_subprocess(cmd, cwd=p.work_dir)


def run_sss_command(fname_in, options, fname_out, host='kasga', port=22,
                    fname_pos=None):
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
    t0 = time()
    remote_in = '~/temp_%s_raw.fif' % t0
    remote_out = '~/temp_%s_raw_sss.fif' % t0
    remote_pos = '~/temp_%s_raw_sss.pos' % t0
    print('Copying file to %s' % host)
    cmd = ['scp', '-P' + port, fname_in, host + ':' + remote_in]
    run_subprocess(cmd, stdout=None, stderr=None)

    if fname_pos is not None:
        options += ' -hp ' + remote_pos

    print('Running maxfilter on %s' % host)
    cmd = ['ssh', '-p', port, host,
           'maxfilter -f ' + remote_in + ' -o ' + remote_out + ' ' + options]
    try:
        run_subprocess(cmd, stdout=None, stderr=None)

        print('Copying result to %s' % fname_out)
        if fname_pos is not None:
            try:
                cmd = ['scp', '-P' + port, host + ':' + remote_pos, fname_pos]
                run_subprocess(cmd, stdout=None, stderr=None)
            except Exception:
                pass
        cmd = ['scp', '-P' + port, host + ':' + remote_out, fname_out]
        run_subprocess(cmd, stdout=None, stderr=None)
    finally:
        print('Cleaning up %s' % host)
        files = [remote_in, remote_out]
        files += [remote_pos] if fname_pos is not None else []
        cmd = ['ssh', '-p', port, host, 'rm -f ' + ' '.join(files)]
        try:
            run_subprocess(cmd, stdout=None, stderr=None)
        except Exception:
            pass


def run_sss_positions(fname_in, fname_out, host='kasga', opts='', port=22):
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
        The SSH/scp host to run the command on.
    opts : str
        Additional command-line options to pass to MaxFilter.
    """
    # let's make sure we can actually write where we want
    if not op.isfile(fname_in):
        raise IOError('input file not found: %s' % fname_in)
    if not op.isdir(op.dirname(op.abspath(fname_out))):
        raise IOError('output directory for output file does not exist')
    fnames_in = [fname_in]
    for ii in range(1, 11):
        next_name = op.splitext(fname_in)[0] + '-%s' % ii + '.fif'
        if op.isfile(next_name):
            fnames_in.append(next_name)
        else:
            break
    port = str(int(port))
    t0 = time()
    remote_ins = ['~/' + op.basename(fname) for fname in fnames_in]
    remote_out = '~/temp_%s_raw_quat.fif' % t0
    remote_hp = '~/temp_%s_hp.txt' % t0
    print('  Copying file to %s' % host)
    cmd = ['scp', '-P' + port] + fnames_in + [host + ':~/']
    run_subprocess(cmd, stdout=None, stderr=None)

    print('  Running maxfilter on %s' % host)
    cmd = ['ssh', '-p', port, host,
           'maxfilter -f ' + remote_ins[0] + ' -o ' + remote_out +
           ' -headpos -format short -hp ' + remote_hp + ' ' + opts]
    run_subprocess(cmd)

    print('  Copying result to %s' % fname_out)
    cmd = ['scp', '-P' + port, host + ':' + remote_hp, fname_out]
    run_subprocess(cmd)

    print('  Cleaning up %s' % host)
    cmd = ['ssh', '-p', port, host, 'rm -f %s %s %s'
           % (' '.join(remote_ins), remote_hp, remote_out)]
    run_subprocess(cmd)


def run_sss_localy(p, subjects, run_indices):
    """Run SSS locally using maxwell filter in python

    .. warning:: Automatic bad channel detection is not currently implemented.
                 It is critical to mark bad channels before running Maxwell
                 filtering, so data should be inspected and marked accordingly
                 prior to running this algorithm.

    .. warning:: Not all features of Elekta MaxFilter™ are currently
                 implemented (see Notes). Maxwell filtering in mne-python
                 is not designed for clinical use.

    Notes
    -----
    Compared to Elekta's MaxFilter™ software, Maxwwell filtering in mne-python
    does not require/support the following arguments:
        -format
        -hpicons
        -autobad
        -force
        -movecomp

    """
    cal_file = op.join(op.dirname(op.dirname(__file__)), 'sss_cal.dat')
    ct_file = op.join(op.dirname(op.dirname(__file__)), 'ct_sparse.fif')
    if p.tsss_dur:
        st_duration = p.tsss_duration
    else:
        st_duration = None
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Preprocessing subject %g/%g (%s).'
                  % (si + 1, len(subjects), subj))
        sss_dir = op.join(p.work_dir, subj, p.sss_dir)
        if not op.isdir(sss_dir):
            os.mkdir(sss_dir)
        # Create SSP projection vectors after marking bad channels
        raw_names = get_raw_fnames(p, subj, 'raw', False, False,
                                   run_indices[si])
        names_out = get_raw_fnames(p, subj, 'sss', False, False,
                                   run_indices[si])
        prebad_file = _prebad(p, subj)
        for ii, (r, o) in enumerate(zip(raw_names, names_out)):
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')
            raw = _raw_LRFCP(raw_names, p.proj_sfreq, None, None, p.n_jobs_fir,
                             p.n_jobs_resample, list(), prebad_file, p.disp_files,
                             method='fft', filter_length=p.filter_length,
                             apply_proj=False, force_bads=True,
                             l_trans=p.hp_trans, h_trans=p.lp_trans, allow_maxshield=True)
            # apply maxwell filter
            if p.sss_origin is 'head':
                _, origin, _ = fit_sphere_to_headshape(raw.info)
            else:
                origin = p.sss_origin

            if p.trans_to is 'median':
                trans_to = op.join(raw_dir, subj + '_median_pos.fif')
                if not op.isfile(trans_to):
                    calc_median_hp(p, subj, trans_to, run_indices[si])
            elif isinstance(p.trans_to, (list, np.ndarray)):
                trans_to = p.trans_to
            else:
                trans_to = None

            raw_sss = maxwell_filter(raw, origin=origin, int_order=p.int_order, ext_order=p.ext_order,
                                     calibration=cal_file,
                                     cross_talk=ct_file,
                                     st_correlation=p.st_correlation, st_duration=st_duration,
                                     destination=trans_to)
            raw_sss.save(o, overwrite=True, buffer_size_sec=None)


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
    with warnings.catch_warnings(record=True):
        raw = Raw(fname, allow_maxshield=True)
    orig_events = find_events(raw, stim_channel='STI101', shortest_event=0)
    events = list()
    for ch in range(1, 9):
        ev = find_events(raw, stim_channel='STI00%d' % ch)
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
            assert isinstance(structurals[si], str)
            assert isinstance(dates[si], tuple) and len(dates[si]) == 3
            assert all([isinstance(d, int) for d in dates[si]])
            anon = dict(first_name=subj, last_name=structurals[si],
                        birthday=dates[si])
        else:
            anon = None
        fix_eeg_channels(names, anon)


def get_fsaverage_medial_vertices(concatenate=True):
    """Returns fsaverage medial wall vertex numbers

    These refer to the standard fsaverage source space
    (with vertices from 0 to 2*10242-1).

    Parameters
    ----------
    concatenate : bool
        If True, the medial wall vertices from the right hemisphere
        will be shifted by 10242 and concatenated to those of the left.
        Useful when treating the source space as a single entity.

    Returns
    -------
    vertices : list of array, or array
        The medial wall vertices.
    """
    label_dir = op.join(get_config('SUBJECTS_DIR'), 'fsaverage', 'label')
    lh = read_label(op.join(label_dir, 'lh.Medial_wall.label'))
    rh = read_label(op.join(label_dir, 'rh.Medial_wall.label'))
    if concatenate is True:
        return np.concatenate((lh.vertices[lh.vertices < 10242],
                               rh.vertices[rh.vertices < 10242] + 10242))
    else:
        return [lh.vertices, rh.vertices]


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
        raise RuntimeError('in_names must have same length as in_numbers')
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
            raw = Raw(raw_fname, preload=False)
            first_samps.append(raw._first_samps[0])
            last_samps.append(raw._last_samps[-1])
        # read in raw files
        raw = [Raw(fname, preload=False) for fname in raw_names]
        _fix_raw_eog_cals(raw, raw_names)  # EOG epoch scales might be bad!
        raw = concatenate_raws(raw)

        # read in events
        events = [read_events(fname) for fname in
                  get_event_fnames(p, subj, run_indices[si])]
        events = concatenate_events(events, first_samps, last_samps)
        # do time adjustment
        t_adj = np.zeros((1, 3), dtype='int')
        t_adj[0, 0] = np.round(-p.t_adjust * raw.info['sfreq']).astype(int)
        events = events.astype(int) + t_adj
        new_sfreq = raw.info['sfreq'] / decim[si]
        if p.disp_files:
            print('    Epoching data (decim=%s -> sfreq=%s Hz).'
                  % (decim[si], new_sfreq))
        if new_sfreq not in sfreqs:
            if len(sfreqs) > 0:
                warnings.warn('resulting new sampling frequency %s not equal '
                              'to previous values %s' % (new_sfreq, sfreqs))
            sfreqs.add(new_sfreq)
        use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat, raw)
        epochs = Epochs(raw, events, event_id=old_dict, tmin=p.tmin,
                        tmax=p.tmax, baseline=_get_baseline(p),
                        reject=use_reject, flat=use_flat, proj='delayed',
                        preload=True, decim=decim[si], on_missing=p.on_missing,
                        reject_tmin=p.reject_tmin, reject_tmax=p.reject_tmax)
        del raw
        drop_logs.append(epochs.drop_log)
        ch_namess.append(epochs.ch_names)
        # only kept trials that were not dropped
        sfreq = epochs.info['sfreq']
        epochs_fnames, evoked_fnames = get_epochs_evokeds_fnames(p, subj,
                                                                 analyses)
        mat_file, fif_file = epochs_fnames
        # now deal with conditions to save evoked
        if p.disp_files:
            print('    Saving evoked data to disk.')
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
                    e.equalize_event_counts(in_names_match, copy=False)

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
                evokeds.append(e[name].average())
                evokeds.append(e[name].standard_error())
            write_evokeds(fn, evokeds)
            if p.disp_files:
                print('      Analysis "%s": %s epochs / condition'
                      % (analysis, evokeds[0].nave))

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
            print('  Subject %s. ' % subj)
        inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        if not op.isdir(inv_dir):
            os.mkdir(inv_dir)
        make_erm_inv = len(p.runs_empty) > 0

        # Shouldn't matter which raw file we use
        raw_fname = get_raw_fnames(p, subj, 'pca', True, False,
                                   run_indices[si])[0]
        raw = Raw(raw_fname)
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
            fwd = read_forward_solution(fwd_name, surf_ori=True)
            looses = [None]
            tags = [p.inv_free_tag]
            fixeds = [False]
            depths = [0.8]
            if fwd['src'][0]['type'] == 'surf':
                looses += [None, 0.2]
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
                                                loose=l, depth=d, fixed=x)
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
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        if not op.isdir(fwd_dir):
            os.mkdir(fwd_dir)
        raw_fname = get_raw_fnames(p, subj, 'sss', False, False,
                                   run_indices[si])[0]
        info = read_info(raw_fname)

        subjects_dir = get_config('SUBJECTS_DIR')
        if structurals[si] is None:  # spherical case
            # create spherical BEM
            bem = make_sphere_model('auto', 'auto', info, verbose=False)
            # create source space
            sphere = np.concatenate((bem['r0'], [bem['layers'][0]['rad']]))
            sphere *= 1000.  # to mm
            src = setup_volume_source_space(subj, None, pos=7., sphere=sphere,
                                            mindist=1.)
            trans = {'from': FIFF.FIFFV_COORD_HEAD,
                     'to': FIFF.FIFFV_COORD_MRI,
                     'trans': np.eye(4)}
            bem_type = 'spherical model'
        else:
            trans = op.join(p.work_dir, subj, p.trans_dir, subj + '-trans.fif')
            if not op.isfile(trans):
                trans = op.join(p.work_dir, subj, p.trans_dir,
                                subj + '-trans_head2mri.txt')
                if not op.isfile(trans):
                    raise IOError('Unable to find head<->MRI trans file')
            src = op.join(subjects_dir, structurals[si], 'bem',
                          structurals[si] + '-oct-6-src.fif')
            if not op.isfile(src):
                print('  Creating source space for %s...' % subj)
                setup_source_space(structurals[si], src, 'oct6',
                                   n_jobs=p.n_jobs)
            bem = op.join(subjects_dir, structurals[si], 'bem',
                          structurals[si] + '-' + p.bem_type + '-bem-sol.fif')
            bem_type = ('%s-layer BEM' %
                        len(read_bem_solution(bem, verbose=False)['surfs']))
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
            make_forward_solution(info, trans, src, bem,
                                  fname=fwd_name, n_jobs=p.n_jobs,
                                  mindist=p.fwd_mindist, overwrite=True)


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
            raw = Raw(empty_fif, preload=True)
            use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat, raw)
            picks = pick_types(raw.info, meg=True, eeg=False, exclude='bads')
            cov = compute_raw_covariance(raw, reject=use_reject,
                                         flat=use_flat, picks=picks)
            write_cov(empty_cov_name, cov)

        # Make evoked covariances
        for inv_name, inv_run in zip(p.inv_names, p.inv_runs):
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
                raws.append(Raw(raw_fname, preload=False))
                first_samps.append(raws[-1]._first_samps[0])
                last_samps.append(raws[-1]._last_samps[-1])
            _fix_raw_eog_cals(raws, raw_fnames)  # safe b/c cov only needs MEEG
            raw = concatenate_raws(raws)
            events = [read_events(e) for e in eve_fnames]
            old_count = sum(len(e) for e in events)
            events = [p.pick_events_cov(e) for e in events]
            new_count = sum(len(e) for e in events)
            if new_count != old_count:
                print('  Using %s instead of %s original events for '
                      'covariance calculation' % (new_count, old_count))
            events = concatenate_events(events, first_samps,
                                        last_samps)
            use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat, raw)
            epochs = Epochs(raw, events, event_id=None, tmin=p.bmin,
                            tmax=p.bmax, baseline=(None, None), proj=False,
                            reject=use_reject, flat=use_flat, preload=True)
            epochs.pick_types(meg=True, eeg=True, exclude=[])
            cov_name = op.join(cov_dir, safe_inserter(inv_name, subj) +
                               ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
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


# noinspection PyPep8Naming
def _raw_LRFCP(raw_names, sfreq, l_freq, h_freq, n_jobs, n_jobs_resample,
               projs, bad_file, disp_files=False, method='fft',
               filter_length=32768, apply_proj=True, preload=True,
               force_bads=False, l_trans=0.5, h_trans=0.5, allow_maxshield=False):
    """Helper to load, filter, concatenate, then project raw files
    """
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    if disp_files:
        print('    Loading and filtering %d files.' % len(raw_names))
    raw = list()
    for rn in raw_names:
        r = Raw(rn, preload=True, allow_maxshield=allow_maxshield)
        r.load_bad_channels(bad_file, force=force_bads)
        if sfreq is not None:
            with warnings.catch_warnings(record=True):  # resamp of stim ch
                r.resample(sfreq, n_jobs=n_jobs_resample)
        if l_freq is not None or h_freq is not None:
            r.filter(l_freq=l_freq, h_freq=h_freq, picks=None,
                     n_jobs=n_jobs, method=method,
                     filter_length=filter_length,
                     l_trans_bandwidth=l_trans, h_trans_bandwidth=h_trans)
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
        if isinstance(p.auto_bad, float):
            print('    Creating bad channel file, marking bad channels:\n'
                  '        %s' % bad_file)
            if not op.isdir(bad_dir):
                os.mkdir(bad_dir)
            # do autobad
            raw = _raw_LRFCP(raw_names, p.proj_sfreq, None, None, p.n_jobs_fir,
                             p.n_jobs_resample, list(), None, p.disp_files,
                             method='fft', filter_length=p.filter_length,
                             apply_proj=False, force_bads=False,
                             l_trans=p.hp_trans, h_trans=p.lp_trans)
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
                      '{0:.0f}% trials dropped:\n'.format(p.auto_bad))
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
        raw_orig = _raw_LRFCP(pre_list, p.proj_sfreq, None, bad_file,
                              p.n_jobs_fir, p.n_jobs_resample, projs, bad_file,
                              p.disp_files, method='fft',
                              filter_length=p.filter_length, force_bads=False,
                              l_trans=p.hp_trans, h_trans=p.lp_trans)

        # Apply any user-supplied extra projectors
        if p.proj_extra is not None:
            if p.disp_files:
                print('    Adding extra projectors from "%s".' % p.proj_extra)
            extra_proj = op.join(pca_dir, p.proj_extra)
            projs = read_proj(extra_proj)

        # Calculate and apply ERM projectors
        if any(proj_nums[2]):
            if len(empty_names) >= 1:
                if p.disp_files:
                    print('    Computing continuous projectors using ERM.')
                # Use empty room(s), but processed the same way
                raw = _raw_LRFCP(empty_names, p.proj_sfreq, None, None,
                                 p.n_jobs_fir, p.n_jobs_resample, projs,
                                 bad_file, p.disp_files, method='fft',
                                 filter_length=p.filter_length,
                                 force_bads=True,
                                 l_trans=p.hp_trans, h_trans=p.lp_trans)
            else:
                if p.disp_files:
                    print('    Computing continuous projectors using data.')
                raw = raw_orig.copy()
            raw.filter(None, p.cont_lp, n_jobs=p.n_jobs_fir, method='fft',
                       filter_length=p.filter_length)
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
                       method='fft', filter_length=p.filter_length)
            raw.add_proj(projs)
            raw.apply_proj()
            pr, ecg_events = \
                compute_proj_ecg(raw, n_grad=proj_nums[0][0],
                                 n_jobs=p.n_jobs_mkl,
                                 n_mag=proj_nums[0][1], n_eeg=proj_nums[0][2],
                                 tmin=ecg_t_lims[0], tmax=ecg_t_lims[1],
                                 l_freq=None, h_freq=None, no_proj=True,
                                 qrs_threshold='auto', ch_name=p.ecg_channel,
                                 reject=p.ssp_ecg_reject)
            if ecg_events.shape[0] >= 20:
                write_events(ecg_eve, ecg_events)
                write_proj(ecg_proj, pr)
                projs.extend(pr)
            else:
                warnings.warn('Only %d ECG events!' % ecg_events.shape[0])
            del raw

        # Next calculate and apply the EOG projectors
        if any(proj_nums[1]):
            if p.disp_files:
                print('    Computing EOG projectors.')
            raw = raw_orig.copy()
            raw.filter(eog_f_lims[0], eog_f_lims[1], n_jobs=p.n_jobs_fir,
                       method='fft', filter_length=p.filter_length)
            raw.add_proj(projs)
            raw.apply_proj()
            pr, eog_events = \
                compute_proj_eog(raw, n_grad=proj_nums[1][0],
                                 n_jobs=p.n_jobs_mkl,
                                 n_mag=proj_nums[1][1], n_eeg=proj_nums[1][2],
                                 tmin=eog_t_lims[0], tmax=eog_t_lims[1],
                                 l_freq=None, h_freq=None, no_proj=True,
                                 ch_name=p.eog_channel,
                                 reject=p.ssp_eog_reject)
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
        raw_orig.filter(p.hp_cut, p.lp_cut, n_jobs=p.n_jobs_fir, method='fft',
                        filter_length=p.filter_length,
                        l_trans_bandwidth=p.hp_trans,
                        h_trans_bandwidth=p.lp_trans)
        raw_orig.add_proj(projs)
        raw_orig.apply_proj()
        # now let's epoch with 1-sec windows to look for DQs
        events = fixed_len_events(p, raw_orig)
        use_reject, use_flat = _restrict_reject_flat(p.reject, p.flat,
                                                     raw_orig)
        epochs = Epochs(raw_orig, events, None, p.tmin, p.tmax, preload=False,
                        baseline=_get_baseline(p), reject=use_reject,
                        flat=use_flat, proj=True)
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
        if len(erm_in) > 0:
            for ii, (r, o) in enumerate(zip(erm_in, erm_out)):
                if p.disp_files:
                    print('    Processing erm file %d/%d.'
                          % (ii + 1, len(erm_in)))
            raw = _raw_LRFCP(r, None, p.hp_cut, p.lp_cut, p.n_jobs_fir,
                             p.n_jobs_resample, projs, bad_file,
                             disp_files=False, method='fft', apply_proj=False,
                             filter_length=p.filter_length, force_bads=True,
                             l_trans=p.hp_trans, h_trans=p.lp_trans)
            raw.save(o, overwrite=True, buffer_size_sec=None)
        for ii, (r, o) in enumerate(zip(names_in, names_out)):
            if p.disp_files:
                print('    Processing file %d/%d.'
                      % (ii + 1, len(names_in)))
            raw = _raw_LRFCP(r, None, p.hp_cut, p.lp_cut, p.n_jobs_fir,
                             p.n_jobs_resample, projs, bad_file,
                             disp_files=False, method='fft', apply_proj=False,
                             filter_length=p.filter_length, force_bads=False,
                             l_trans=p.hp_trans, h_trans=p.lp_trans)
            raw.save(o, overwrite=True, buffer_size_sec=None)
        # look at raw_clean for ExG events
        if p.plot_raw:
            _viz_raw_ssp_events(p, subj, run_indices[si])


class FakeEpochs():
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


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming
def anova_time(X):
    """A mass-univariate two-way ANOVA (with time as a co-variate)

    Parameters
    ----------
    X : array
        X should have the following dimensions:
            subjects x (2 conditions x N time points) x spatial locations
        This then calculates the paired t-values at each spatial location
        using time as a co-variate.

    Returns
    -------
    t : array
        t-values from the contrast, has the same length as the number of
        spatial locations.
    p : array
        Corresponding p values of the contrast.
    dof : int
        Degrees of freedom, with conservative Greenhouse-Geisser
        correction based on the number of time points (n_time - 1).
    """
    import patsy
    from scipy import linalg, stats

    n_subjects, n_nested, n_sources = X.shape
    n_time = n_nested / 2
    # Turn Y into (2 x n_time x n_subjects) x n_sources
    Y = np.sqrt(np.reshape(X, (2 * n_time * n_subjects, n_sources), order='F'))
    cv, tv, sv = np.meshgrid(np.arange(2.0), np.arange(n_time),
                             np.arange(n_subjects), indexing='ij')
    dmat = patsy.dmatrix('C(cv) + C(tv) + C(sv)',
                         dict(sv=sv.ravel(), tv=tv.ravel(), cv=cv.ravel()))
    c = np.zeros((1, dmat.shape[1]))
    c[0, 1] = 1  # Contrast for just picking up condition difference
    b = np.dot(linalg.pinv(dmat), Y)
    d = Y - np.dot(dmat, b)
    r = dmat.shape[0] - np.linalg.matrix_rank(dmat)
    R = np.diag(np.dot(d.T, d))[:, np.newaxis] / r
    e = np.sqrt(R * np.dot(c, np.dot(linalg.pinv(np.dot(dmat.T, dmat)), c.T)))
    t = (np.dot(c, b) / e.T).T
    dof = r / (n_time - 1)  # Greenhouse-Geisser correction to the DOF
    p = np.sign(t) * 2 * stats.t.cdf(-abs(t), dof)
    return t, p, dof


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
                     method='fft', filter_length=p.filter_length,
                     force_bads=False, l_trans=p.hp_trans, h_trans=p.lp_trans)
    raw.plot(events=ev, event_color=colors)


def gen_html_report(p, subjects, structurals, run_indices=None,
                    raw=True, evoked=True, cov=True, trans=True, epochs=True):
    """Generates HTML reports"""
    types = ['filtered raw', 'evoked', 'covariance', 'trans', 'epochs']
    texts = ['*fil%d*sss.fif' % p.lp_cut, '*ave.fif',
             '*cov.fif', '*trans.fif', '*epo.fif']
    if run_indices is None:
        run_indices = [None] * len(subjects)
    for si, subj in enumerate(subjects):
        bools = [raw, evoked, cov, trans, epochs]
        path = op.join(p.work_dir, subj)
        files = []
        for ii, (b, text) in enumerate(zip(bools, texts)):
            files.append(glob.glob(path + '/*/' + text))
        bools = [False if not f else b for f, b in zip(files, bools)]
        missing = ', '.join([t for t, b in zip(types, bools) if not b])
        if len(missing) > 0:
            print('    For %s no reports generated for:\n        %s'
                  % (subj, missing))
        patterns = [t for t, b in zip(texts, bools) if b]
        fnames = get_raw_fnames(p, subj, 'pca', False, False, run_indices[si])
        if not fnames:
            raise RuntimeError('Could not find any processed files for '
                               'reporting.')
        info_fname = op.join(path, fnames[0])
        struc = structurals[si]
        report = Report(info_fname=info_fname, subject=struc)
        report.parse_folder(data_path=path, mri_decim=10, n_jobs=p.n_jobs,
                            pattern=patterns)
        report_fname = get_report_fnames(p, subj)[0]
        report.save(report_fname, open_browser=False, overwrite=True)


def plot_raw_psd(p, subjects, run_indices=None, tmin=0., fmin=2, n_fft=2048):
    """Plot data power for all available raw data files for a subject

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    run_indices : array-like | None
        Run indices to include.
    tmin : float
        Time in sec for beginning fft (defaults to 0)
    fmin : float
        Lower frequency edge for PSD (defaults to 2Hz)
    n_fft : int
        Number of points in the FFT.

    Notes
    -----
    tmax for psd set to last time point in raw data. fmax set
    to acquisition low pass cut off for raw and sss files, and
    low pass cut off in analysis parameters for pca file. n_fft
    set to default value from mne-python.
    """
    if run_indices is None:
        run_indices = [None] * len(subjects)
    for si, subj in enumerate(subjects):
        for file_type in ['raw', 'sss', 'pca']:
            fname = get_raw_fnames(p, subj, file_type, False, False,
                                   run_indices[si])
            if len(fname) < 1:
                warnings.warn('Unable to find %s data file.' % file_type)
            with warnings.catch_warnings(record=True):
                raw = Raw(fname, preload=True, allow_maxshield=True)
            if file_type == 'pca':
                fmax = p.lp_cut
            else:
                fmax = raw.info['lowpass'] + 50
            raw.plot_psd(tmin=tmin, tmax=raw.times[-1], fmin=fmin,
                         fmax=fmax, n_fft=n_fft,
                         n_jobs=p.n_jobs, proj=False, ax=None, color=(0, 0, 1),
                         picks=None, show=False)
            plt.savefig(fname[0][:-4] + '_psd.png')
            plt.close()


def _prebad(p, subj):
    """Helper for locating file containing bad channels during acq"""
    prebad_file = op.join(p.work_dir, subj, p.raw_dir, subj + '_prebad.txt')
    if not op.isfile(prebad_file):  # SSS prebad file
        raise RuntimeError('Could not find SSS prebad file: %s'
                               % prebad_file)
    return prebad_file

