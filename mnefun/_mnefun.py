import os
import os.path as op
import numpy as np
from scipy import io as spio
import warnings
from shutil import move
import subprocess
import re
import glob

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
from mne.io import Raw, concatenate_raws, read_info
from mne.io.pick import pick_types_forward, pick_types
from mne.cov import regularize
from mne.minimum_norm import write_inverse_operator
from mne.layouts import make_eeg_layout
from mne.viz import plot_drop_log


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
                 filter_length=32768, drop_thresh=1, fname_style='new',
                 epochs_type=('fif', 'mat'), fwd_mindist=2.0):
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
        fname_style : str
            Style of filenames. 'old' is SSS_FIF, RAW_FIF, etc.
            'new' is sss_fif, raw_fif, etc.
        epochs_type : str | list
            Can be 'fif', 'mat', or a list containing both.
        fwd_mindist : float
            Minimum distance for sources in the brain from the skull in order
            for them to be included in the forward solution source space.

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
        self.orig_dir_tag = None
        self.raw_dir_tag = None
        self.fif_tag = None
        self.inv_tag = None
        self.keep_orig = True
        self.n_jobs = n_jobs
        self.n_jobs_mkl = n_jobs_mkl
        self.n_jobs_fir = 'cuda'  # Jobs when using method='fft'
        self.n_jobs_resample = n_jobs_resample
        self.filter_length = filter_length
        self.bad_tag = None
        self.cont_lp = 5
        self.lp_cut = lp_cut
        self.mne_root = os.getenv('MNE_ROOT')
        self.disp_files = True
        self.proj_sfreq = proj_sfreq
        self.decim = decim
        self.drop_thresh = drop_thresh
        self.fname_style = fname_style
        if isinstance(epochs_type, string_types):
            epochs_type = (epochs_type,)
        if not all([t in ('mat', 'fif') for t in epochs_type]):
            raise ValueError('All entries in "epochs_type" must be "mat" '
                             'or "fif"')
        self.epochs_type = epochs_type
        self.fwd_mindist = fwd_mindist


def fix_eeg_files(p, subjects, structurals=None, dates=None, verbose=True):
    """Reorder EEG channels based on UW cap setup and params

    Reorders any SSS and non-SSS files it can find based on params.
    It will try to fix both types of files by using p.orig_dir_tag
    and p.

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
        raw_dir = op.join(p.work_dir, subj, p.orig_dir_tag)

        # Create SSP projection vectors after marking bad channels
        raw_names = [op.join(raw_dir, safe_inserter(r, subj) + p.fif_tag)
                     for r in p.run_names]
        raw_names += [op.join(raw_dir, safe_inserter(r, subj) + p.fif_tag)
                      for r in p.runs_empty]
        if p.extra_dir_tag is not None:
            raw_dir = op.join(p.work_dir, subj, p.extra_dir_tag)
            raw_names += [op.join(raw_dir, safe_inserter(r, subj) +
                                  p.extra_fif_tag) for r in p.run_names]
            raw_names += [op.join(raw_dir, safe_inserter(r, subj) +
                                  p.extra_fif_tag) for r in p.runs_empty]
        # Now let's make sure we only run files that actually exist
        names = []
        for name in raw_names:
            if op.isfile(name):
                names += [name]
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
            for file in move_files:
                move(file, file + '.orig')
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
                out_numbers, must_match, match_fun=None):
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
    match_fun : function | None
        If None, standard matching will be performed. If a function,
        must_match will be ignored, and ``match_fun`` will be called
        to equalize event counts.
    """
    in_names = np.asanyarray(in_names)
    old_dict = dict()
    for n, e in zip(in_names, in_numbers):
        old_dict[n] = e

    fif_extra = ('_allclean_fil%d' % p.lp_cut) + p.fif_tag
    n_runs = len(p.run_names)
    ch_namess = list()
    drop_logs = list()
    for subj in subjects:
        if p.disp_files:
            print('  Loading raw files for subject %s.' % subj)
        pca_dir = op.join(p.work_dir, subj, p.raw_dir_tag)
        lst_dir = op.join(p.work_dir, subj, p.list_dir)
        epochs_dir = op.join(p.work_dir, subj, p.epochs_dir)
        if not op.isdir(epochs_dir):
            os.mkdir(epochs_dir)
        evoked_dir = op.join(p.work_dir, subj, p.inverse_dir)
        if not op.isdir(evoked_dir):
            os.mkdir(evoked_dir)

        # read in raw files
        raw_names = [op.join(pca_dir, safe_inserter(r, subj) + fif_extra)
                     for r in p.run_names]
        raw = Raw(raw_names, preload=False)

        # read in events
        events = [read_events(op.join(lst_dir, 'ALL_' +
                  safe_inserter(p.run_names[ri], subj) + '.lst'))
                  for ri in range(n_runs)]
        events = concatenate_events(events, raw._first_samps, raw._last_samps)
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
            if match_fun is None:
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
                e = match_fun(epochs.copy(), analysis, nn,
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
    meg_out_flags = ['-meg', '-meg-eeg', '-eeg']
    meg_bools = [True, True, False]
    eeg_bools = [False, True, True]

    fif_extra = ('_allclean_fil%d' % p.lp_cut)
    for subj in subjects:
        if p.disp_files:
            print('  Subject %s. ' % (subj))
        inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        pca_dir = op.join(p.work_dir, subj, p.raw_dir_tag)
        if not op.isdir(inv_dir):
            os.mkdir(inv_dir)
        make_erm_inv = len(p.runs_empty) > 0
        if make_erm_inv:
            cov_name = op.join(cov_dir, safe_inserter(p.runs_empty[0], subj)
                               + ('_allclean_fil%d' % p.lp_cut)
                               + p.inv_tag + '-cov.fif')
            empty_cov = read_cov(cov_name)
        else:
            cov_name = op.join(cov_dir, safe_inserter(subj, subj)
                               + ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
        for name in p.inv_names:
            s_name = safe_inserter(name, subj)
            temp_name = s_name + ('-%d' % p.lp_cut) + p.inv_tag
            fwd_name = op.join(fwd_dir, s_name + p.inv_tag + '-fwd.fif')
            fwd = read_forward_solution(fwd_name, surf_ori=True)
            # Shouldn't matter which raw file we use
            raw_fname = op.join(pca_dir, safe_inserter(p.run_names[0], subj)
                                + fif_extra + p.fif_tag)
            raw = Raw(raw_fname)

            cov = read_cov(cov_name)
            cov_reg = regularize(cov, raw.info)
            if make_erm_inv:
                empty_cov_reg = regularize(empty_cov, raw.info)
            for f, m, e in zip(meg_out_flags, meg_bools, eeg_bools):
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
        raw_dir = op.join(p.work_dir, subj, p.orig_dir_tag)
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        if not op.isdir(fwd_dir):
            os.mkdir(fwd_dir)

        subjects_dir = get_config('SUBJECTS_DIR')
        mri_file = op.join(p.work_dir, subj, p.trans_dir, subj + '-trans.fif')
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
                           structural + '-5120-5120-5120-bem-sol.fif')
        if p.data_transformed:
            for ii, (inv_name, inv_run) in enumerate(zip(p.inv_names,
                                                         p.inv_runs)):
                s_name = safe_inserter(p.run_names[inv_run[0]], subj)
                raw_name = op.join(raw_dir, s_name + p.fif_tag)
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
                raw_name = op.join(raw_dir, s_name + p.fif_tag)
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
    fif_extra = ('_allclean_fil%d' % p.lp_cut)
    for subj in subjects:
        pca_dir = op.join(p.work_dir, subj, p.raw_dir_tag)
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        lst_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(cov_dir):
            os.mkdir(cov_dir)

        # Make empty room cov
        if p.runs_empty:
            if len(p.runs_empty) > 1:
                raise ValueError('Too many empty rooms; undefined output!')
            new_run = safe_inserter(p.runs_empty[0], subj)
            empty_cov_name = op.join(cov_dir, new_run + fif_extra
                                     + p.inv_tag + '-cov.fif')
            empty_fif = op.join(pca_dir, new_run + fif_extra + p.fif_tag)
            raw = Raw(empty_fif, preload=True)
            cov = compute_raw_data_covariance(raw, reject=p.reject,
                                              flat=p.flat)
            write_cov(empty_cov_name, cov)

        # Make evoked covariances
        rn = p.run_names
        for inv_name, inv_run in zip(p.inv_names, p.inv_runs):
            raw_fnames = [op.join(pca_dir, safe_inserter(rn[ir], subj)
                                  + fif_extra + p.fif_tag)
                          for ir in inv_run]
            raw = Raw(raw_fnames, preload=False)
            e_names = [op.join(lst_dir, 'ALL_' + safe_inserter(rn[ir], subj)
                       + '.lst') for ir in inv_run]
            events = [read_events(e) for e in e_names]
            events = concatenate_events(events, raw._first_samps,
                                        raw._last_samps)
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
        raw_dir = op.join(p.work_dir, subj, p.orig_dir_tag)
        pca_dir = op.join(p.work_dir, subj, p.raw_dir_tag)
        bad_dir = op.join(p.work_dir, subj, p.bad_dir_tag)

        # Create SSP projection vectors after marking bad channels
        raw_names = [op.join(raw_dir, safe_inserter(r, subj) + p.fif_tag)
                     for r in p.run_names]
        empty_names = [op.join(raw_dir, safe_inserter(r, subj) + p.fif_tag)
                       for r in p.runs_empty]
        for r in raw_names + empty_names:
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')

        bad_file = op.join(bad_dir, 'bad_ch_' + subj + p.bad_tag)
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
        raw_orig = _raw_LRFCP(pre_list, p.proj_sfreq, None, None, p.n_jobs_fir,
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
                                 qrs_threshold=0.9)
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
        dur = p.tmax - p.tmin
        events = make_fixed_length_events(raw_orig, 1, duration=dur)
        epochs = Epochs(raw_orig, events, None, p.tmin, p.tmax, preload=False,
                        baseline=(p.bmin, p.bmax), reject=p.reject,
                        flat=p.flat, proj=False)
        epochs.drop_bad_epochs()
        drop_logs.append(epochs.drop_log)
        del raw_orig
        del epochs
    return drop_logs


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
        raw_dir = op.join(p.work_dir, subj, p.orig_dir_tag)
        pca_dir = op.join(p.work_dir, subj, p.raw_dir_tag)
        raw_names = [op.join(raw_dir, safe_inserter(r, subj) + p.fif_tag)
                     for r in p.run_names]
        emp_names = [op.join(raw_dir, safe_inserter(r, subj) + p.fif_tag)
                     for r in p.runs_empty]
        names_out = [op.join(pca_dir, safe_inserter(r, subj)
                     + '_allclean_fil%d' % p.lp_cut + p.fif_tag)
                     for r in p.run_names + p.runs_empty]
        bad_dir = op.join(p.work_dir, subj, p.bad_dir_tag)
        bad_file = op.join(bad_dir, 'bad_ch_' + subj + p.bad_tag)
        bad_file = None if not op.isfile(bad_file) else bad_file
        all_proj = op.join(pca_dir, 'preproc_all-proj.fif')
        projs = read_proj(all_proj)
        for ii, (r, o) in enumerate(zip(raw_names + emp_names, names_out)):
            if p.disp_files:
                print('    Processing file %d/%d.'
                      % (ii + 1, len(raw_names) + len(emp_names)))
            raw = _raw_LRFCP(r, None, None, p.lp_cut, p.n_jobs_fir,
                             p.n_jobs_resample, projs, bad_file,
                             disp_files=False, method='fft', apply_proj=False,
                             filter_length=p.filter_length)
            raw.save(o, overwrite=True)


def lst_read(raw_path, stim_channel='STI101'):
    """Wrapper for find_events that defaults to UW stim_channel STI101

    Parameters
    ----------
    raw_path : str
        Path to raw file to extract events from.
    stim_channel : str
        Stim channel to use. Defaults to STI101.

    Returns
    -------
    events : array
        Nx4 array of events.
    """
    raw = Raw(raw_path, allow_maxshield=True)
    lst_in = find_events(raw, stim_channel=stim_channel)
    raw.close()
    return lst_in


def _mne_head_sphere(in_fif, out_pos):
    """Calculate sphere from head digitization for SSS

    Parameters
    ----------
    in_fif : str
        FIF file to use.
    out_pos : str
        Filename to save the head position to.
    """
    raw = Raw(in_fif, allow_maxshield=True)
    radius, origin_head, origin_device = fit_sphere_to_headshape(raw.info)
    raw.close()
    out_string = ''.join(['%0.0f ' % np.round(number)
                          for number in origin_head])
    if out_pos:
        f = open(out_pos, 'w')
        f.write(out_string)
        f.close()


def calc_head_centers(p, subjects):
    """Calculate sphere locations from head digitizations for SSS

    Saves head positions to a file.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    """
    if p.extra_dir_tag is None:
        dir_tag = p.orig_dir_tag
        fif_tag = p.fif_tag
    else:
        dir_tag = p.extra_dir_tag
        fif_tag = p.extra_fif_tag
    pout_dir = op.join(p.work_dir, 'SSSPrep')
    if not op.isdir(pout_dir):
        os.mkdir(pout_dir)
    for si in range(len(subjects)):
        for ri in range(len(p.run_names)):
            new_run = safe_inserter(p.run_names[ri], subjects[si])
            in_fif = op.join(p.work_dir, subjects[si], dir_tag,
                             new_run + fif_tag)
            out_pos = op.join(pout_dir, new_run + '_pos.txt')
            if op.isfile(out_pos):
                os.remove(out_pos)
            _mne_head_sphere(in_fif, out_pos)


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
        in_fif = op.join(p.work_dir, subjects[si], p.orig_dir_tag,
                         new_run + p.fif_tag)
        out_lout = op.join(lout_dir, subjects[si] + '_eeg.lout')
        if op.isfile(out_lout):
            os.remove(out_lout)

        raw = Raw(in_fif)
        make_eeg_layout(raw.info).save(out_lout)
        raw.close()


def make_standard_tags(p, use_sss=True, data_transformed=None):
    """Make standard parameter tags

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    use_sss : bool
        Sets parameters correctly based on whether or not SSS is used.
    data_transformed : bool | None
        Indicate whether or not data has been transformed to a common
        coordinate frame. If None, will be True for SSS and False for non-SSS.

    Returns
    -------
    p : instance of Parameters
        The modified parameters (modified inplace).
    """
    if data_transformed is None:
        data_transformed = True if use_sss else False
    p.data_transformed = data_transformed
    if use_sss:
        if p.fname_style == 'new':
            p.orig_dir_tag = 'sss_fif'
            p.raw_dir_tag = 'sss_pca_fif'
            p.inv_tag = '-sss'
            p.extra_dir_tag = 'raw_fif'
            p.epochs_dir = 'epochs'
            p.epochs_tag = '-epo'
            p.cov_dir = 'covariance'
            p.inverse_dir = 'inverse'
            p.forward_dir = 'forward'
            p.list_dir = 'lists'
            p.trans_dir = 'trans'
            p.bad_dir_tag = 'bads'
            p.inv_fixed_tag = '-fixed'
            p.inv_erm_tag = '-erm'
            p.eq_tag = 'eq'
        else:
            p.orig_dir_tag = 'SSS_FIF'
            p.raw_dir_tag = 'SSS_PCA_FIF'
            p.inv_tag = '-SSS'
            p.extra_dir_tag = 'RAW_FIF'
            p.epochs_dir = 'Epochs'
            p.epochs_tag = '_Epochs'
            p.cov_dir = 'Cov'
            p.inverse_dir = 'Inverse'
            p.forward_dir = 'Forward'
            p.list_dir = 'LST'
            p.trans_dir = 'TRANS'
            p.bad_dir_tag = 'BAD_CH'
            p.inv_fixed_tag = '-Fixed'
            p.inv_erm_tag = '-ERM'
            p.eq_tag = 'EQ'
        p.fif_tag = '_raw_sss.fif'
        p.bad_tag = '_post-sss.txt'
        p.keep_orig = False
        # This is used by fix_eeg_channels to fix original files
        p.extra_fif_tag = '_raw.fif'
    else:
        if p.fname_style == 'new':
            p.orig_dir_tag = 'raw_fif'
            p.raw_dir_tag = 'pca_fif'
        else:
            p.orig_dir_tag = 'RAW_FIF'
            p.raw_dir_tag = 'PCA_FIF'
        p.fif_tag = '_raw.fif'
        p.inv_tag = ''
        p.bad_tag = '_pre-sss.txt'
        p.keep_orig = True
        p.extra_dir_tag = None
        p.extra_fif_tag = None
    return p


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
