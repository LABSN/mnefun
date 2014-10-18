# -*- coding: utf-8 -*-
# Copyright (c) 2014, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import print_function

import os
import os.path as op
import numpy as np
from scipy import io as spio
import warnings
from shutil import move, copy2
import subprocess
import re
import glob
import collections
import matplotlib.pyplot as plt
from time import time
from numpy.testing import assert_allclose

from mne import (compute_proj_raw, make_fixed_length_events, Epochs,
                 find_events, read_events, write_events, concatenate_events,
                 read_cov, write_cov, read_forward_solution,
                 compute_raw_data_covariance, compute_covariance,
                 write_proj, read_proj, setup_source_space,
                 make_forward_solution, average_forward_solutions,
                 write_forward_solution, get_config, write_evokeds,
                 add_source_space_distances, write_source_spaces)
from mne.preprocessing.ssp import compute_proj_ecg, compute_proj_eog
from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.minimum_norm import make_inverse_operator
from mne.label import read_label
from mne.epochs import combine_event_ids
from mne.io import Raw, concatenate_raws, read_info, write_info
from mne.io.base import _quart_to_rot
from mne.io.pick import pick_types_forward, pick_types
from mne.cov import regularize
from mne.minimum_norm import write_inverse_operator
from mne.layouts import make_eeg_layout
from mne.viz import plot_drop_log
from mne.utils import run_subprocess


# python2/3 conversions
try:
    string_types = basestring  # noqa
except Exception:
    string_types = str

try:
    from functools import reduce
except Exception:
    pass


class Params(object):
    def __init__(self, tmin=None, tmax=None, t_adjust=0, bmin=-0.2, bmax=0.0,
                 n_jobs=6, lp_cut=55, decim=5, proj_sfreq=None, n_jobs_mkl=1,
                 n_jobs_fir='cuda', n_jobs_resample='cuda',
                 filter_length=32768, drop_thresh=1,
                 epochs_type='fif', fwd_mindist=2.0,
                 bem_type='5120-5120-5120', auto_bad=None,
                 ecg_channel='ECG063', plot_raw=False, match_fun=None):
        """Make a useful parameter structure

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
        ecg_channel : str
            The channel to use to detect ECG events.
        plot_raw : bool
            If True, plot the raw files with the ECG/EOG events overlaid.
        match_fun : function | None
            If None, standard matching will be performed. If a function,
            must_match will be ignored, and ``match_fun`` will be called
            to equalize event counts.

        Returns
        -------
        params : instance of Params
            The parameters to use.
        """
        self.reject = dict(eog=np.inf, grad=1500e-13, mag=5000e-15, eeg=150e-6)
        self.flat = dict(eog=-1, grad=1e-13, mag=1e-15, eeg=1e-6)
        self.tmin = tmin
        self.tmax = tmax
        self.t_adjust = t_adjust
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
        self.mne_root = os.getenv('MNE_ROOT')
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
        self.ecg_channel = ecg_channel
        self.plot_raw = plot_raw
        self.translate_positions = True

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
        self.inv_erm_tag = '-erm'
        self.eq_tag = 'eq'
        self.sss_fif_tag = '_raw_sss.fif'
        self.bad_tag = '_post-sss.txt'
        self.keep_orig = False
        # This is used by fix_eeg_channels to fix original files
        self.raw_fif_tag = '_raw.fif'

    @property
    def pca_extra(self):
        return '_allclean_fil%d' % self.lp_cut

    @property
    def pca_fif_tag(self):
        return self.pca_extra + self.sss_fif_tag


def do_processing(p, fetch_raw=False, push_raw=False, do_sss=False,
                  fetch_sss=False, do_score=False, do_ch_fix=False,
                  gen_ssp=False, apply_ssp=False, gen_covs=False,
                  gen_fwd=False, gen_inv=False, write_epochs=False):
    """Do M/EEG data processing

    fetch_raw : bool
        Fetch raw recording files from acquisition machine.
    push_raw : bool
        Push raw recording files to SSS workstation.
    do_sss : bool
        Run SSS remotely on SSS workstation.
    fetch_sss : bool
        Fetch SSS files from SSS workstation.
    do_score : bool
        Do scoring.
    do_ch_fix : bool
        Fix channel ordering.
    gen_ssp : bool
        Generate SSP vectors.
    apply_ssp : bool
        Apply SSP vectors and filtering.
    gen_covs : bool
        Generate covariances.
    gen_fwd : bool
        Generate forward solutions.
    get_inv : bool
        Generate inverses.
    write_epochs : bool
        Write epochs to disk.
    """
    # Generate requested things
    bools = [fetch_raw, push_raw, do_sss, fetch_sss, do_score, do_ch_fix,
             gen_ssp, apply_ssp, gen_covs, gen_fwd, gen_inv, write_epochs]
    texts = ['Pulling raw files from acquisition machine',
             'Pushing raw files to remote workstation',
             'Running SSS on remote workstation',
             'Pulling SSS files from remote workstation',
             'Scoring subjects', 'Fixing EEG order', 'Preprocessing files',
             'Applying preprocessing', 'Generating covariances',
             'Generating forward models', 'Generating inverse solutions',
             'Doing epoch EQ/DQ']
    funcs = [fetch_raw_files, push_raw_files, run_sss_remotely,
             fetch_sss_files,
             p.score, fix_eeg_files, do_preprocessing_combined,
             apply_preprocessing_combined, gen_covariances,
             gen_forwards, gen_inverses, save_epochs]
    assert len(bools) == len(texts) == len(funcs)

    sinds = p.subject_indices
    subjects = np.array(p.subjects)[sinds].tolist()
    structurals = np.array(p.structurals)[sinds].tolist()
    dates = [tuple([int(dd) for dd in d]) for d in np.array(p.dates)[sinds]]

    outs = [None] * len(bools)
    for ii, (b, text, func) in enumerate(zip(bools, texts, funcs)):
        if b:
            t0 = time()
            print(text + '. ')
            if func == fix_eeg_files:
                outs[ii] = func(p, subjects, structurals, dates)
            elif func == gen_forwards:
                outs[ii] = func(p, subjects, structurals)
            elif func == save_epochs:
                outs[ii] = func(p, subjects, p.in_names, p.in_numbers,
                                p.analyses, p.out_names, p.out_numbers,
                                p.must_match)
            else:
                outs[ii] = func(p, subjects)
            print('  (' + timestring(time() - t0) + ')')
    print("Done")


def fetch_raw_files(p, subjects):
    """Fetch remote raw recording files (only designed for *nix platforms)"""
    for subj in subjects:
        print('  Checking for proper remote filenames for %s...' % subj)
        subj_dir = op.join(p.work_dir, subj)
        if not op.isdir(subj_dir):
            os.mkdir(subj_dir)
        raw_dir = op.join(subj_dir, p.raw_dir)
        if not op.isdir(raw_dir):
            os.mkdir(raw_dir)
        finder_stem = 'find %s ' % p.acq_dir
        # build remote raw file finder
        fnames = _get_raw_names(p, subj, 'raw', True)
        assert len(fnames) > 0
        fetch = list()
        regex = re.compile(r"-?[0-9]*.fif")
        for f in fnames:
            fetch.append('.*' + op.basename(f[:-4] + regex.pattern))
        finder = finder_stem + ' -o '.join(["-type f -regex %s" % fname
                                            for fname in fetch])
        stdout_ = run_subprocess(['ssh', p.acq_ssh, finder])[0]
        remote_fnames = [x.strip() for x in stdout_.splitlines()]
        assert all(fname.startswith(p.acq_dir) for fname in remote_fnames)
        remote_fnames = [fname[len(p.acq_dir) + 1:] for fname in remote_fnames]
        want = set(op.basename(fname) for fname in fnames)
        got = set([op.basename(fname) for fname in remote_fnames])
        if want != got.intersection(want):
            raise RuntimeError('Could not find all files.\n'
                               'Wanted: %s\nGot: %s' % (want, got.intersection(want)))
        if len(remote_fnames) != len(fnames):
            warnings.warn('Found more files than expected on remote server')
            print('\nLikely split files were found. Please confirm results.')
        print('  Pulling %s files for %s...' % (len(remote_fnames), subj))
        cmd = ['rsync', '-ave', 'ssh', '--prune-empty-dirs', '--partial',
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


def calc_median_hp(p, subj, out_file):
    """Calculate median head position"""
    print('    Estimating median head position for %s... ' % subj)
    raw_files = _get_raw_names(p, subj, 'raw', False)
    ts = []
    qs = []
    out_info = None
    for fname in raw_files:
        info = read_info(fname)
        trans = info['dev_head_t']['trans']
        ts.append(trans[:3, 3])
        m = trans[:3, :3]
        qw = np.sqrt(1. + m[0, 0] + m[1, 1] + m[2, 2]) / 2.
        # make sure we are orthogonal and special
        assert_allclose(np.dot(m, m.T), np.eye(3), atol=1e-5)
        assert_allclose([qw], [1.], atol=5e-2)
        qs.append([(m[2, 1] - m[1, 2]) / (4 * qw),
                   (m[0, 2] - m[2, 0]) / (4 * qw),
                   (m[1, 0] - m[0, 1]) / (4 * qw)])
        assert_allclose(_quart_to_rot(np.array([qs[-1]]))[0],
                        m, rtol=1e-5, atol=1e-5)
        if out_info is None:
            out_info = info
    assert out_info is not None  # at least one raw file
    if len(raw_files) == 1:  # only one head position
        t = ts[0]
        rot = m
    else:
        t = np.median(np.array(ts), axis=0)
        rot = np.median(_quart_to_rot(np.array(qs)), axis=0)
    trans = np.r_[np.c_[rot, t[:, np.newaxis]],
                  np.array([0, 0, 0, 1], t.dtype)[np.newaxis, :]]
    dev_head_t = {'to': 4, 'from': 1, 'trans': trans}
    info = dict(dev_head_t=dev_head_t)
    for key in ('dig', 'chs', 'nchan', 'sfreq', 'lowpass', 'highpass',
                'projs', 'comps', 'bads', 'acq_pars', 'acq_stim',
                'ctf_head_t'):
        info[key] = out_info[key]
    write_info(out_file, info)


def push_raw_files(p, subjects):
    """Push raw files to SSS workstation"""
    if len(subjects) == 0:
        return
    print('  Pushing raw files to SSS workstation...')
    # do all copies at once to avoid multiple logins
    copy2(op.join(op.dirname(__file__), 'run_sss.sh'), p.work_dir)
    includes = ['--include', '/run_sss.sh']
    for subj in subjects:
        subj_dir = op.join(p.work_dir, subj)
        raw_dir = op.join(subj_dir, p.raw_dir)

        out_pos = op.join(raw_dir, subj + '_center.txt')
        if not op.isfile(out_pos):
            print('    Determining head center for %s... ' % subj, end='')
            in_fif = op.join(raw_dir,
                             safe_inserter(p.run_names[0], subj)
                             + p.raw_fif_tag)
            origin_head = fit_sphere_to_headshape(read_info(in_fif))[1]
            out_string = ' '.join(['%0.0f' % np.round(number)
                                   for number in origin_head])
            with open(out_pos, 'w') as fid:
                fid.write(out_string)
            print('(%s)' % out_string)

        med_pos = op.join(raw_dir, subj + '_median_pos.fif')
        if not op.isfile(med_pos):
            calc_median_hp(p, subj, med_pos)
        root = op.sep + subj
        raw_root = op.join(root, p.raw_dir)
        includes += ['--include', root, '--include', raw_root,
                     '--include', op.join(raw_root, op.basename(out_pos)),
                     '--include', op.join(raw_root, op.basename(med_pos))]
        prebad_file = op.join(raw_dir, subj + '_prebad.txt')
        if op.isfile(prebad_file):  # SSS prebad file
            includes += ['--include',
                         op.join(raw_root, op.basename(prebad_file))]
        # build local raw file finder
        finder_stem = 'find %s ' % raw_dir
        fnames = _get_raw_names(p, subj, 'raw', True)
        assert len(fnames) > 0
        fetch = list()
        regex = re.compile(r"-?[0-9]*.fif")
        for f in fnames:
            fetch.append('.*' + op.basename(f[:-4] + regex.pattern))
        finder = finder_stem + ' -o '.join(["-type f -regex %s" % fname
                                            for fname in fetch])
        stdout_ = run_subprocess(finder.split())[0]
        fnames = [x.strip() for x in stdout_.splitlines()]
        for fname in fnames:
            assert op.isfile(op.join(fname)), fname
            includes += ['--include', op.join(raw_root, op.basename(fname))]
    assert ' ' not in p.sws_dir
    assert ' ' not in p.sws_ssh
    cmd = ['rsync', '-ave', 'ssh', '--partial'] + includes + ['--exclude', '*']
    cmd += ['.', '%s:%s' % (p.sws_ssh, op.join(p.sws_dir, ''))]
    run_subprocess(cmd, cwd=p.work_dir)


def run_sss_remotely(p, subjects):
    """Run SSS preprocessing remotely (only designed for *nix platforms)"""
    for subj in subjects:
        s = 'Remote output for %s:' % subj
        print('-' * len(s))
        print(s)
        print('-' * len(s))
        files = ':'.join([op.basename(f)
                          for f in _get_raw_names(p, subj, 'raw', False)])
        erm = ':'.join([op.basename(f)
                        for f in _get_raw_names(p, subj, 'raw', 'only')])
        erm = ' --erm ' + erm if len(erm) > 0 else ''
        run_sss = (op.join(p.sws_dir, 'run_sss.sh') +
                   ' --subject ' + subj + ' --files ' + files + erm)
        cmd = ['ssh', p.sws_ssh, run_sss]
        run_subprocess(cmd, stdout=None, stderr=None)
        print('-' * 70, end='\n\n')


def fetch_sss_files(p, subjects):
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
    cmd = ['rsync', '-ave', 'ssh', '--partial'] + includes + ['--exclude', '*']
    cmd += ['%s:%s' % (p.sws_ssh, op.join(p.sws_dir, '*')), '.']
    run_subprocess(cmd, cwd=p.work_dir)


def extract_expyfun_events(fname):
    """Extract expyfun-style serial-coded events from file

    Parameters
    ----------
    fname : str
        Filename to use.

    Returns
    -------
    events : array
        Array of events of shape (N, 3), re-coded such that 1 triggers
        are renamed according to their binary expyfun representation.
    presses : list of arrays
        List of all press events that occurred between each one
        trigger. Each array has shape (N_presses, 2).
    orig_events : array
        Original events array.
    """
    # Read events
    raw = Raw(fname, allow_maxshield=True)
    orig_events = find_events(raw, stim_channel='STI101')
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
        these = events[breaks[ti + 1]:breaks[ti + 2], 2]
        resp = these[these > 8]
        resp = np.log2(resp) - 3
        resps.append(resp)

        # look at trial coding, double-check trial type (pre-1 trig)
        these = events[breaks[ti + 0]:breaks[ti + 1], 2]
        serials = these[np.logical_and(these >= 4, these <= 8)]
        en = np.sum(2 ** np.arange(len(serials))[::-1] * (serials == 8)) + 1
        event_nums.append(en)

    these_events = events[aud_idx]
    these_events[:, 2] = event_nums
    return these_events, resps, orig_events


def _get_raw_names(p, subj, which, erm):
    """Helper to get raw names"""
    assert which in ('sss', 'raw', 'pca')
    if which == 'sss':
        raw_dir = op.join(p.work_dir, subj, p.sss_dir)
        tag = p.sss_fif_tag
    elif which == 'raw':
        raw_dir = op.join(p.work_dir, subj, p.raw_dir)
        tag = p.raw_fif_tag
    elif which == 'pca':
        raw_dir = op.join(p.work_dir, subj, p.pca_dir)
        tag = p.pca_extra + p.sss_fif_tag
    if erm == 'only':
        use = p.runs_empty
    elif erm:
        use = p.run_names + p.runs_empty
    else:
        use = p.run_names
    return [op.join(raw_dir, safe_inserter(r, subj) + tag) for r in use]


def fix_eeg_files(p, subjects, structurals=None, dates=None, verbose=True):
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
    """
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Fixing subject %g/%g.' % (si + 1, len(subjects)))
        raw_names = _get_raw_names(p, subj, 'sss', True)
        # Now let's make sure we only run files that actually exist
        names = [name for name in raw_names if op.isfile(name)]
        if structurals is not None and dates is not None:
            assert isinstance(structurals[si], str)
            assert isinstance(dates[si], tuple) and len(dates[si]) == 3
            assert all([isinstance(d, int) for d in dates[si]])
            anon = dict(first_name=subj, last_name=structurals[si],
                        birthday=dates[si])
        else:
            anon = None
        fix_eeg_channels(names, anon)


def fix_eeg_channels(raw_files, anon=None, verbose=True):
    """Reorder EEG channels based on UW cap setup

    Parameters
    ----------
    raw_files : list of str | str
        The raw file name(s) to reorder, if it has not been done yet.
    anon : dict | None
        If None, no anonymization is done. If dict, should have the following:
        ``['first_name', 'last_name', 'birthday']``. Names should be strings,
        while birthday should be a tuple of ints (year, month, day).
    verbose : bool
        If True, print whether or not the files were modified.
    """
    order = np.array([1, 2, 3, 5, 6, 7, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 30,
                      31, 32, 33, 34, 35, 36, 37, 38,
                      41, 42, 43, 44, 45, 46, 47, 48, 49,
                      51, 52, 54, 55, 56, 57, 58, 60,
                      39, 29, 18, 4, 8, 28, 40, 59, 50, 53]) - 1
    assert len(order) == 60
    write_key = 'LABSN_EEG_REORDER:' + ','.join([str(o) for o in order])
    if anon is None:
        anon_key = ''
    else:
        anon_key = ';anonymized'

    # do some type checking
    if not isinstance(raw_files, list):
        raw_files = [raw_files]

    # actually do the reordering
    for ri, raw_file in enumerate(raw_files):
        raw = Raw(raw_file, preload=False, allow_maxshield=True)
        picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])
        if not len(picks) == len(order):
            raise RuntimeError('Incorrect number of EEG channels found (%i)'
                               % len(picks))
        need_reorder = (write_key not in raw.info['description'])
        need_anon = (anon is not None and
                     (anon_key not in raw.info['description']))
        if need_anon or need_reorder:
            to_do = []
            if need_reorder:
                to_do += ['reordering']
            if need_anon:
                to_do += ['anonymizing']
            to_do = ' & '.join(to_do)
            # Now we need to preorder
            if verbose:
                print('    Making a backup and %s file %i' % (to_do, ri + 1))
            raw = Raw(raw_file, preload=True, allow_maxshield=True)
            # rename split files if any
            regex = re.compile("-*.fif")
            split_files = glob.glob(raw_file[:-4] + regex.pattern)
            move_files = [raw_file] + split_files
            for f in move_files:
                move(f, f + '.orig')
            if need_reorder:
                raw._data[picks, :] = raw._data[picks, :][order]
            if need_anon:
                raw.info['subject_info'].update(anon)
            raw.info['description'] = write_key + anon_key
            raw.save(raw_file, format=raw.orig_format, overwrite=True)
        else:
            if verbose:
                print('    File %i already corrected' % (ri + 1))


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


def save_epochs(p, subjects, in_names, in_numbers, analyses, out_names,
                out_numbers, must_match):
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
    """
    in_names = np.asanyarray(in_names)
    old_dict = dict()
    for n, e in zip(in_names, in_numbers):
        old_dict[n] = e

    n_runs = len(p.run_names)
    ch_namess = list()
    drop_logs = list()
    for subj in subjects:
        if p.disp_files:
            print('  Loading raw files for subject %s.' % subj)
        lst_dir = op.join(p.work_dir, subj, p.list_dir)
        epochs_dir = op.join(p.work_dir, subj, p.epochs_dir)
        if not op.isdir(epochs_dir):
            os.mkdir(epochs_dir)
        evoked_dir = op.join(p.work_dir, subj, p.inverse_dir)
        if not op.isdir(evoked_dir):
            os.mkdir(evoked_dir)

        # read in raw files
        raw_names = _get_raw_names(p, subj, 'pca', False)
        # read in events
        first_samps = []
        last_samps = []
        for raw_fname in raw_names:
            raw = Raw(raw_fname, preload=False)
            first_samps.append(raw._first_samps[0])
            last_samps.append(raw._last_samps[-1])
        # read in raw files
        raw = Raw(raw_names, preload=False)

        # read in events
        events = [read_events(op.join(lst_dir, 'ALL_' +
                                      safe_inserter(p.run_names[ri], subj) +
                                      '-eve.lst'))
                  for ri in range(n_runs)]
        events = concatenate_events(events, first_samps, last_samps)
        # do time adjustment
        t_adj = np.zeros((1, 3), dtype='int')
        t_adj[0, 0] = np.round(-p.t_adjust * raw.info['sfreq']).astype(int)
        events = events.astype(int) + t_adj
        if p.disp_files:
            print('    Epoching data.')
        epochs = Epochs(raw, events, event_id=old_dict, tmin=p.tmin,
                        tmax=p.tmax, baseline=(p.bmin, p.bmax),
                        reject=p.reject, flat=p.flat, proj=True,
                        preload=True, decim=p.decim)
        del raw
        drop_logs.append(epochs.drop_log)
        ch_namess.append(epochs.ch_names)
        # only kept trials that were not dropped
        sfreq = epochs.info['sfreq']
        mat_file = op.join(epochs_dir, 'All_%d' % p.lp_cut +
                           p.inv_tag + '_' + subj + p.epochs_tag + '.mat')
        fif_file = op.join(epochs_dir, 'All_%d' % p.lp_cut +
                           p.inv_tag + '_' + subj + p.epochs_tag + '.fif')
        # now deal with conditions to save evoked
        if p.disp_files:
            print('    Saving evoked data to disk.')
        for analysis, names, numbers, match in zip(analyses, out_names,
                                                   out_numbers, must_match):
            # do matching
            numbers = np.asanyarray(numbers)
            nn = numbers[numbers >= 0]
            new_numbers = np.unique(numbers[numbers >= 0])
            in_names_match = in_names[match]
            if not len(new_numbers) == len(names):
                raise ValueError('out_numbers length must match out_names '
                                 'length for analysis %s' % analysis)
            if p.match_fun is None:
                # first, equalize trial counts (this will make a copy)
                if len(in_names_match) > 1:
                    e = epochs.equalize_event_counts(in_names_match)[0]
                else:
                    e = epochs.copy()

                # second, collapse types
                for num, name in zip(new_numbers, names):
                    combine_event_ids(e, in_names[num == numbers], {name: num},
                                      copy=False)
            else:  # custom matching
                e = p.match_fun(epochs.copy(), analysis, nn,
                                in_names_match, names)

            # now make evoked for each out type
            evokeds = list()
            for name in names:
                evokeds.append(e[name].average())
                evokeds.append(e[name].standard_error())
            fn = '%s_%d%s_%s_%s-ave.fif' % (analysis, p.lp_cut, p.inv_tag,
                                            p.eq_tag, subj)
            write_evokeds(op.join(evoked_dir, fn), evokeds)
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


def gen_inverses(p, subjects, use_old_rank=False):
    """Generate inverses

    Can only complete successfully following forward solution
    calculation and covariance estimation.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    """
    for subj in subjects:
        meg, eeg, meeg = _channels_types(p, subj)
        if meeg:
            out_flags = ['-meg', '-meg-eeg', '-eeg']
            meg_bools = [True, True, False]
            eeg_bools = [False, True, True]
        elif meg:
            out_flags = ['-meg']
            meg_bools = [True]
            eeg_bools = [False]
        elif eeg:
            out_flags = ['-eeg']
            meg_bools = [False]
            eeg_bools = [True]
        if p.disp_files:
            print('  Subject %s. ' % subj)
        inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        pca_dir = op.join(p.work_dir, subj, p.pca_dir)
        if not op.isdir(inv_dir):
            os.mkdir(inv_dir)
        make_erm_inv = len(p.runs_empty) > 0
        if make_erm_inv:
            erm_name = op.join(cov_dir, safe_inserter(p.runs_empty[0], subj)
                               + p.pca_extra + p.inv_tag + '-cov.fif')
            empty_cov = read_cov(erm_name)
        for name in p.inv_names:
            s_name = safe_inserter(name, subj)
            temp_name = s_name + ('-%d' % p.lp_cut) + p.inv_tag
            fwd_name = op.join(fwd_dir, s_name + p.inv_tag + '-fwd.fif')
            fwd = read_forward_solution(fwd_name, surf_ori=True)
            # Shouldn't matter which raw file we use
            raw_fname = op.join(pca_dir, safe_inserter(p.run_names[0], subj)
                                + p.pca_fif_tag)
            raw = Raw(raw_fname)

            cov_name = op.join(cov_dir, safe_inserter(name, subj)
                               + ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
            cov = read_cov(cov_name)
            cov_reg = regularize(cov, raw.info)
            if make_erm_inv:
                empty_cov_reg = regularize(empty_cov, raw.info)
            for f, m, e in zip(out_flags, meg_bools, eeg_bools):
                fwd_restricted = pick_types_forward(fwd, meg=m, eeg=e)
                for l, s, x in zip([None, 0.2], [p.inv_fixed_tag, ''],
                                   [True, False]):
                    inv_name = op.join(inv_dir,
                                       temp_name + f + s + '-inv.fif')
                    inv = make_inverse_operator(raw.info, fwd_restricted,
                                                cov_reg, loose=l, depth=0.8,
                                                fixed=x)
                    write_inverse_operator(inv_name, inv)
                    if (not e) and p.runs_empty:
                        inv_name = op.join(inv_dir, temp_name + f
                                           + p.inv_erm_tag + s + '-inv.fif')
                        inv = make_inverse_operator(raw.info, fwd_restricted,
                                                    empty_cov_reg, fixed=x,
                                                    loose=l, depth=0.8)
                        write_inverse_operator(inv_name, inv)


def gen_forwards(p, subjects, structurals):
    """Generate forward solutions

    Can only complete successfully once coregistration is performed
    (usually in mne_analyze).

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    structurals : list of str
        The structural data names for each subject (e.g., ['AKCLEE_101', ...]).
    """
    for subj, structural in zip(subjects, structurals):
        raw_dir = op.join(p.work_dir, subj, p.sss_dir)
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        if not op.isdir(fwd_dir):
            os.mkdir(fwd_dir)

        subjects_dir = get_config('SUBJECTS_DIR')
        mri_file = op.join(p.work_dir, subj, p.trans_dir, subj + '-trans.fif')
        if not op.isfile(mri_file):
            mri_file = op.join(p.work_dir, subj, p.trans_dir, subj + '-trans_head2mri.txt')
        elif not op.isfile(mri_file):
            raise IOError('Unable to find coordinate transformation file')
        src_file = op.join(subjects_dir, structural, 'bem',
                           structural + '-oct-6-src.fif')
        if not op.isfile(src_file):
            print('  Creating source space for %s...' % subj)
            src = setup_source_space(structural, None, 'oct6')
            print('  Adding distances and patch information...')
            add_source_space_distances(src, n_jobs=p.n_jobs)
            write_source_spaces(src_file, src)
            print('  Creating forward solution(s)...')
        bem_file = op.join(subjects_dir, structural, 'bem',
                           structural + '-' + p.bem_type + '-bem-sol.fif')
        if p.translate_positions:
            for ii, (inv_name, inv_run) in enumerate(zip(p.inv_names,
                                                         p.inv_runs)):
                s_name = safe_inserter(p.run_names[inv_run[0]], subj)
                raw_name = op.join(raw_dir, s_name + p.sss_fif_tag)
                info = read_info(raw_name)
                fwd_name = op.join(fwd_dir, safe_inserter(inv_name, subj)
                                   + p.inv_tag + '-fwd.fif')
                make_forward_solution(info, mri_file, src_file, bem_file,
                                      fname=fwd_name, n_jobs=p.n_jobs,
                                      mindist=p.fwd_mindist, overwrite=True)
        else:
            # Legacy code for when runs are in different positions
            fwds = list()
            for ri, run_name in enumerate(p.run_names):
                s_name = safe_inserter(run_name, subj)
                raw_name = op.join(raw_dir, s_name + p.sss_fif_tag)
                info = read_info(raw_name)
                fwd_name = op.join(fwd_dir, s_name + p.inv_tag + '-fwd.fif')
                fwd = make_forward_solution(info, mri_file, src_file, bem_file,
                                            fname=fwd_name, n_jobs=p.n_jobs,
                                            overwrite=True)
                fwds.append(fwd)

            for ii, (inv_name, inv_run) in enumerate(zip(p.inv_names,
                                                         p.inv_runs)):
                fwds_use = [f for fi, f in enumerate(fwds) if fi in inv_run]
                fwd_name = op.join(fwd_dir, safe_inserter(inv_name, subj)
                                   + p.inv_tag + '-fwd.fif')
                fwd_ave = average_forward_solutions(fwds_use)
                write_forward_solution(fwd_name, fwd_ave, overwrite=True)


def gen_covariances(p, subjects):
    """Generate forward solutions

    Can only complete successfully once preprocessing is performed.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    """
    for subj in subjects:
        pca_dir = op.join(p.work_dir, subj, p.pca_dir)
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        lst_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(cov_dir):
            os.mkdir(cov_dir)

        # Make empty room cov
        if p.runs_empty:
            if len(p.runs_empty) > 1:
                raise ValueError('Too many empty rooms; undefined output!')
            new_run = safe_inserter(p.runs_empty[0], subj)
            empty_cov_name = op.join(cov_dir, new_run + p.pca_extra
                                     + p.inv_tag + '-cov.fif')
            empty_fif = op.join(pca_dir, new_run + p.pca_fif_tag)
            raw = Raw(empty_fif, preload=True)
            cov = compute_raw_data_covariance(raw, reject=p.reject,
                                              flat=p.flat)
            write_cov(empty_cov_name, cov)

        # Make evoked covariances
        rn = p.run_names
        for inv_name, inv_run in zip(p.inv_names, p.inv_runs):
            raw_fnames = [op.join(pca_dir, safe_inserter(rn[ir], subj)
                                  + p.pca_fif_tag)
                          for ir in inv_run]
            first_samps = []
            last_samps = []
            for raw_fname in raw_fnames:
                raw = Raw(raw_fname, preload=False)
                first_samps.append(raw._first_samps[0])
                last_samps.append(raw._last_samps[-1])
            raw = Raw(raw_fnames, preload=False)
            e_names = [op.join(lst_dir, 'ALL_' + safe_inserter(rn[ir], subj)
                               + '-eve.lst') for ir in inv_run]
            events = [read_events(e) for e in e_names]
            events = concatenate_events(events, first_samps,
                                        last_samps)
            epochs = Epochs(raw, events, event_id=None, tmin=p.bmin,
                            tmax=p.bmax, baseline=(None, None), preload=True)
            cov_name = op.join(cov_dir, safe_inserter(inv_name, subj)
                               + ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
            cov = compute_covariance(epochs)
            write_cov(cov_name, cov)


def safe_inserter(string, inserter):
    """Helper to insert a subject name into a string if %s is present

    Parameters
    ----------
    string : str
        String to fill.

    inserter : str
        The string to put in ``string`` if ``'%s'`` is present.

    Returns
    -------
    string : str
        The modified string.
    """
    if '%s' in string:
        string = string % inserter
    return string


# noinspection PyPep8Naming
def _raw_LRFCP(raw_names, sfreq, l_freq, h_freq, n_jobs, n_jobs_resample,
               projs, bad_file, disp_files=False, method='fft',
               filter_length=32768, apply_proj=True, preload=True):
    """Helper to load, filter, concatenate, then project raw files
    """
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    if disp_files:
        print('    Loading and filtering %d files.' % len(raw_names))
    raw = list()
    for rn in raw_names:
        r = Raw(rn, preload=True)
        r.load_bad_channels(bad_file)
        if sfreq is not None:
            r.resample(sfreq, n_jobs=n_jobs_resample)
        if l_freq is not None or h_freq is not None:
            r.filter(l_freq=l_freq, h_freq=h_freq, picks=None,
                     n_jobs=n_jobs, method=method,
                     filter_length=filter_length)
        raw.append(r)
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


def do_preprocessing_combined(p, subjects):
    """Do preprocessing on all raw files together

    Calculates projection vectors to use to clean data.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    """
    drop_logs = list()
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Preprocessing subject %g/%g.' % (si + 1, len(subjects)))
        pca_dir = op.join(p.work_dir, subj, p.pca_dir)
        bad_dir = op.join(p.work_dir, subj, p.bad_dir)

        # Create SSP projection vectors after marking bad channels
        raw_names = _get_raw_names(p, subj, 'sss', False)
        empty_names = _get_raw_names(p, subj, 'sss', 'only')
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
                             method='fft', filter_length=p.filter_length, apply_proj=False)
            events = fixed_len_events(p, raw)
            # do not mark eog channels bad
            meg, eeg = _channels_types(p, subj)[:2]
            picks = pick_types(raw.info, meg=meg, eeg=eeg, eog=False, exclude=[])
            assert type(p.auto_bad_reject) and type(p.auto_bad_flat) == dict
            epochs = Epochs(raw, events, picks=picks, event_id=None, tmin=p.tmin,
                            tmax=p.tmax, baseline=(p.bmin, p.bmax),
                            reject=p.auto_bad_reject, flat=p.auto_bad_flat, proj=True,
                            preload=True, decim=0)
            # channel scores from drop log
            scores = collections.Counter([ch for d in epochs.drop_log for ch in d])
            ch_names = np.array(scores.keys())
            # channel scores expressed as percentile and rank ordered
            counts = 100 * np.array(scores.values(), dtype=float) / len(epochs.drop_log)
            order = np.flipud(np.argsort(counts))
            # boolean array masking out channels with less than % epochs dropped
            mask = counts[order] > p.auto_bad
            badchs = ch_names[order[mask]]
            if len(badchs) >= 1:
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
        reject = dict(grad=np.inf, mag=np.inf, eeg=np.inf, eog=np.inf)
        raw_orig = _raw_LRFCP(pre_list, p.proj_sfreq, None, bad_file, p.n_jobs_fir,
                              p.n_jobs_resample, projs, bad_file, p.disp_files,
                              method='fft', filter_length=p.filter_length)

        if any(proj_nums[2]):
            if len(empty_names) >= 1:
                if p.disp_files:
                    print('    Computing continuous projectors using ERM.')
                # Use empty room(s), but processed the same way
                raw = _raw_LRFCP(empty_names, p.proj_sfreq, None, None,
                                 p.n_jobs_fir, p.n_jobs_resample, projs,
                                 bad_file, p.disp_files, method='fft',
                                 filter_length=p.filter_length)
            else:
                if p.disp_files:
                    print('    Computing continuous projectors using data.')
                raw = raw_orig.copy()
            raw.filter(None, p.cont_lp, n_jobs=p.n_jobs_fir, method='fft',
                       filter_length=p.filter_length)
            raw.apply_proj()
            pr = compute_proj_raw(raw, duration=1, n_grad=proj_nums[2][0],
                                  n_mag=proj_nums[2][1], n_eeg=proj_nums[2][2],
                                  reject=reject, flat=None,
                                  n_jobs=p.n_jobs_mkl)
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
            o = compute_proj_ecg(raw, n_grad=proj_nums[0][0],
                                 n_jobs=p.n_jobs_mkl,
                                 n_mag=proj_nums[0][1], n_eeg=proj_nums[0][2],
                                 tmin=ecg_t_lims[0], tmax=ecg_t_lims[1],
                                 l_freq=None, h_freq=None, no_proj=True,
                                 qrs_threshold='auto', ch_name=p.ecg_channel)
            pr, ecg_events = o
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
            o = compute_proj_eog(raw, n_grad=proj_nums[1][0],
                                 n_jobs=p.n_jobs_mkl,
                                 n_mag=proj_nums[1][1], n_eeg=proj_nums[1][2],
                                 tmin=eog_t_lims[0], tmax=eog_t_lims[1],
                                 l_freq=None, h_freq=None, no_proj=True)
            pr, eog_events = o
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
        raw_orig.filter(None, p.lp_cut, n_jobs=p.n_jobs_fir, method='fft',
                        filter_length=p.filter_length)
        raw_orig.add_proj(projs)
        raw_orig.apply_proj()
        # now let's epoch with 1-sec windows to look for DQs
        events = fixed_len_events(p, raw_orig)
        epochs = Epochs(raw_orig, events, None, p.tmin, p.tmax, preload=False,
                        baseline=(p.bmin, p.bmax), reject=p.reject,
                        flat=p.flat, proj=False)
        epochs.drop_bad_epochs()
        drop_logs.append(epochs.drop_log)
        del raw_orig
        del epochs
    if p.plot_drop_logs:
        for subj, drop_log in zip(subjects, drop_logs):
            plot_drop_log(drop_log, p.drop_thresh, subject=subj)


def apply_preprocessing_combined(p, subjects):
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
    """
    # Now actually save some data
    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('  Applying processing to subject %g/%g.'
                  % (si + 1, len(subjects)))
        pca_dir = op.join(p.work_dir, subj, p.pca_dir)
        names_in = _get_raw_names(p, subj, 'sss', True)
        names_out = _get_raw_names(p, subj, 'pca', True)
        bad_dir = op.join(p.work_dir, subj, p.bad_dir)
        bad_file = op.join(bad_dir, 'bad_ch_' + subj + p.bad_tag)
        bad_file = None if not op.isfile(bad_file) else bad_file
        all_proj = op.join(pca_dir, 'preproc_all-proj.fif')
        projs = read_proj(all_proj)
        for ii, (r, o) in enumerate(zip(names_in, names_out)):
            if p.disp_files:
                print('    Processing file %d/%d.'
                      % (ii + 1, len(names_in)))
            raw = _raw_LRFCP(r, None, None, p.lp_cut, p.n_jobs_fir,
                             p.n_jobs_resample, projs, bad_file,
                             disp_files=False, method='fft', apply_proj=False,
                             filter_length=p.filter_length)
            raw.save(o, overwrite=True)
        # look at raw_clean for ExG events
        if p.plot_raw:
            viz_raw_ssp_events(p, subj)


def gen_layouts(p, subjects):
    """Generate .lout files for each subject

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    """
    lout_dir = op.join(p.mne_root, 'share', 'mne', 'mne_analyze', 'lout')
    for si in range(len(subjects)):
        ri = 1
        new_run = safe_inserter(p.run_names[ri], subjects[si])
        in_fif = op.join(p.work_dir, subjects[si], p.sss_dir,
                         new_run + p.sss_fif_tag)
        out_lout = op.join(lout_dir, subjects[si] + '_eeg.lout')
        if op.isfile(out_lout):
            os.remove(out_lout)

        raw = Raw(in_fif)
        make_eeg_layout(raw.info).save(out_lout)
        raw.close()


class FakeEpochs():
    """Make iterable epoch-like class, convenient for MATLAB transition"""

    def __init__(self, data, ch_names, tmin=-0.2, sfreq=1000.0):
        self._data = data
        self.info = dict(ch_names=ch_names, sfreq=sfreq)
        self.times = np.arange(data.shape[-1]) / sfreq + tmin
        self._current = 0
        self.ch_names = ch_names

    def __iter__(self):
        self._current = 0
        return self

    def next(self):
        if self._current >= len(self._data):
            raise StopIteration
        epoch = self._data[self._current]
        self._current += 1
        return epoch


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
    rediv = lambda ll, b: list(divmod(ll[0], b)) + ll[1:]
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


def viz_raw_ssp_events(p, subj, show=True):
    """Helper to plot filtered cleaned raw trace with ExG events"""
    pca_dir = op.join(p.work_dir, subj, p.raw_dir_tag)
    sss_dir = op.join(p.work_dir, subj, p.orig_dir_tag)
    raw_names = [op.join(sss_dir, safe_inserter(r, subj) + p.fif_tag)
                 for r in p.run_names]
    pre_list = [r for ri, r in enumerate(raw_names)
                if ri in p.get_projs_from]
    all_proj = op.join(pca_dir, 'preproc_all-proj.fif')
    projs = read_proj(all_proj)
    ev_names = [op.join(pca_dir + '/' + ii) for ii in
                ['preproc_ecg-eve.fif', 'preproc_blink-eve.fif']]
    ev = [read_events(e) for e in ev_names]
    assert len(ev) == 2
    ev = np.concatenate((ev[0], ev[1]))
    ev.sort(axis=0)
    raw = _raw_LRFCP(pre_list, p.proj_sfreq, None, None, p.n_jobs_fir,
                     p.n_jobs_resample, projs, None, p.disp_files,
                     method='fft', filter_length=p.filter_length)
    if show:
        raw.plot(events=ev)
        plt.draw()
        plt.show()


def _channels_types(p, subj):
    """Returns bools for MEG, EEG, M/EEG channel types in data info"""
    info = read_info(_get_raw_names(p, subj, 'sss', False)[0])
    meg = len(pick_types(info, meg=True, eeg=False)) > 0
    eeg = len(pick_types(info, meg=False, eeg=True)) > 0
    return meg, eeg, (meg and eeg)
