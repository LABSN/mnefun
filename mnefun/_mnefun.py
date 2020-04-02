# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import json
import os
import os.path as op
import time
import warnings

import numpy as np

from ._utils import get_args

from ._fetching import fetch_raw_files
from ._scoring import default_score
from ._sss_legacy import push_raw_files, fetch_sss_files
from ._sss import run_sss
from ._fix import fix_eeg_files
from ._ssp import do_preprocessing_combined, apply_preprocessing_combined
from ._epoching import save_epochs
from ._cov import gen_covariances
from ._forward import gen_forwards
from ._inverse import gen_inverses
from ._status import print_proc_status
from ._paths import _get_config_file
from ._utils import timestring


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
    """Make a parameter structure for use with `do_processing`.

    See the :ref:`overview` for a description of the options.

    Some params can be set on init, but it's better to set them in
    a YAML file.

    See also
    --------
    do_processing
    read_params
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
        self.eog_thresh = None
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
        self.run_names = []
        self.inv_names = []
        self.inv_runs = []
        self.work_dir = os.getcwd()
        self.n_jobs = n_jobs
        self.n_jobs_mkl = n_jobs_mkl
        self.n_jobs_fir = n_jobs_fir  # Jobs when using method='fir'
        self.n_jobs_resample = n_jobs_resample
        self.filter_length = filter_length
        self.cont_lp = 5.
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
        if isinstance(epochs_type, str):
            epochs_type = [epochs_type]
        if not all([t in ('mat', 'fif') for t in epochs_type]):
            raise ValueError('All entries in "epochs_type" must be "mat" '
                             'or "fif"')
        self.epochs_type = epochs_type
        self.fwd_mindist = fwd_mindist
        self.mf_autobad = False
        self.mf_autobad_type = 'maxfilter'
        self.mf_badlimit = 7
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
        self.cal_file = 'uw'
        self.ct_file = 'uw'
        # SSS denoising params
        self.sss_type = 'maxfilter'
        self.hp_type = 'maxfilter'
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
        self.autoreject_types = ['mag', 'grad', 'eeg']
        self.subjects_dir = None
        self.src_pos = 7.
        self.report_params = dict(
            chpi_snr=True,
            good_hpi_count=True,
            head_movement=True,
            raw_segments=True,
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
        self.coil_gof_limit = 0.98
        self.coil_t_window = 0.2  # default is same as MF
        self.coil_t_step_min = 0.01
        self.proj_ave = False
        self.compute_rank = False
        self.cov_rank = 'full'
        self.force_erm_cov_rank_full = True  # force empty-room inv rank
        self.cov_rank_tol = 1e-6
        self.eog_t_lims = (-0.25, 0.25)
        self.ecg_t_lims = (-0.08, 0.08)
        self.eog_f_lims = (0, 2)
        self.ecg_f_lims = (5, 35)
        self.proj_meg = 'separate'
        self.src = 'oct6'
        self.epochs_prefix = 'All'
        self.reject_epochs_by_annot = True
        self.prebad = dict()
        self.freeze()
        # Read static-able paraws from config file
        _set_static(self)

    @property
    def report(self):  # wrapper
        return self.report_params

    @report.setter
    def report(self, report_params):
        self.report_params = report_params

    @property
    def pca_extra(self):
        return '_allclean_fil%d' % self.lp_cut

    @property
    def pca_fif_tag(self):
        return self.pca_extra + self.sss_fif_tag

    def convert_subjects(self, subj_template, struc_template=None):
        """Convert subject names.

        Parameters
        ----------
        subj_template : str
            Subject template to use.
        struc_template : str
            Structural template to use.
        """
        if struc_template is not None:
            if isinstance(struc_template, str):
                def fun(x):
                    return struc_template % x
            else:
                fun = struc_template
            new = [fun(subj) for subj in self.subjects]
            assert all(isinstance(subj, str) for subj in new)
            self.structurals = new
        if isinstance(subj_template, str):
            def fun(x):
                return subj_template % x
        else:
            fun = subj_template
        new = [fun(subj) for subj in self.subjects]
        assert all(isinstance(subj, str) for subj in new)
        self.subjects = new

    def save(self, fname):
        """Save to a YAML file.

        Parameters
        ----------
        fname : str
            The filename to use. Should end in '.yml'.
        """
        from ._yaml import _write_params
        if not (isinstance(fname, str) and fname.endswith('.yml')):
            raise ValueError('fname should be a str ending with .yml, got %r'
                             % (fname,))
        _write_params(fname, self)


def _set_static(p):
    config_file = _get_config_file()
    key_cast = dict(
        sws_dir=str,
        sws_ssh=str,
        sws_port=int,
    )
    if op.isfile(config_file):
        try:
            with open(config_file, 'rb') as fid:
                config = json.load(fid)
        except Exception as exp:
            raise RuntimeError('Could not parse mnefun config file %s, got '
                               'likely JSON syntax error:\n%s'
                               % (config_file, exp))
        for key, cast in key_cast.items():
            if key in config:
                setattr(p, key, cast(config[key]))
    else:
        warnings.warn(
            'Machine configuration file %s not found, for best compatibility '
            'you should create this file.' % (config_file,))
    for key in key_cast:
        if getattr(p, key, None) is None:
            warnings.warn('Configuration value p.%s was None, remote MaxFilter'
                          ' processing will fail.' % (key,))


def do_processing(p, fetch_raw=False, do_score=False, push_raw=False,
                  do_sss=False, fetch_sss=False, do_ch_fix=False,
                  gen_ssp=False, apply_ssp=False,
                  write_epochs=False, gen_covs=False, gen_fwd=False,
                  gen_inv=False, gen_report=False, print_status=True):
    """Do M/EEG data processing.

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
    from ._report import gen_html_report
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
                 for d in [p.dates[si] for si in sinds]]

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
        run_indices = [None] * len(subjects)
    elif isinstance(run_indices, dict):
        run_indices = [run_indices[subject] for subject in subjects]
    else:
        run_indices = [run_indices[si] for si in sinds]
        assert len(run_indices) == len(subjects)
    run_indices = [np.array(run_idx) if run_idx is not None
                   else np.arange(len(p.run_names)) for run_idx in run_indices]
    assert all(run_idx.ndim == 1 for run_idx in run_indices)
    assert all(np.in1d(r, np.arange(len(p.run_names))).all()
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
            elif func == gen_covariances:
                outs[ii] = func(p, subjects, run_indices, decim)
            else:
                outs[ii] = func(p, subjects, run_indices)
            print('  (' + timestring(time.time() - t0) + ')')
            if p.on_process is not None:
                p.on_process(text, func, outs[ii], p)
    print("Done")
