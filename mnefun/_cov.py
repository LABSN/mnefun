"""Covariance calculation."""

import os
import os.path as op

import numpy as np
from scipy import linalg
from mne import (read_epochs, compute_covariance, write_cov,
                 compute_raw_covariance, Epochs, compute_rank)
from mne.cov import compute_whitener
from mne.externals.h5io import read_hdf5
from mne.io import read_raw_fif
from mne.rank import estimate_rank
from mne.viz import plot_cov

from ._epoching import _concat_resamp_raws
from ._paths import get_epochs_evokeds_fnames, get_raw_fnames, safe_inserter
from ._scoring import _read_events
from ._utils import (get_args, _get_baseline, _restrict_reject_flat,
                     _handle_dict, _handle_decim, _check_reject_annot_regex)


def _compute_rank(p, subj, run_indices):
    """Compute rank of the data."""
    epochs_fnames, _ = get_epochs_evokeds_fnames(p, subj, p.analyses)
    _, fif_file = epochs_fnames
    epochs = read_epochs(fif_file)  # .crop(p.bmin, p.bmax)  maybe someday...?
    meg, eeg = 'meg' in epochs, 'eeg' in epochs
    rank = dict()
    epochs.apply_proj()
    if p.cov_rank_method == 'estimate_rank':
        # old way
        if meg:
            eps = epochs.copy().pick_types(meg=meg, eeg=False)
            eps = eps.get_data().transpose([1, 0, 2])
            eps = eps.reshape(len(eps), -1)
            if 'grad' in epochs and 'mag' in epochs:  # Neuromag
                key = 'meg'
            else:
                key = 'grad' if 'grad' in epochs else 'mag'
            rank[key] = estimate_rank(eps, tol=p.cov_rank_tol)
        if eeg:
            eps = epochs.copy().pick_types(meg=False, eeg=eeg)
            eps = eps.get_data().transpose([1, 0, 2])
            eps = eps.reshape(len(eps), -1)
            rank['eeg'] = estimate_rank(eps, tol=p.cov_rank_tol)
    else:
        assert p.cov_rank_method == 'compute_rank'
        # new way
        rank = compute_rank(epochs, tol=p.cov_rank_tol, tol_kind='relative')
    for k, v in rank.items():
        print(' : %s rank %2d' % (k.upper(), v), end='')
    return rank


def gen_covariances(p, subjects, run_indices, decim):
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
    decim : list of int
        The subject decimations.
    """
    for si, subj in enumerate(subjects):
        print('  Subject %2d/%2d...' % (si + 1, len(subjects)), end='')
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        if not op.isdir(cov_dir):
            os.mkdir(cov_dir)
        has_rank_arg = 'rank' in get_args(compute_covariance)
        kwargs = dict()
        kwargs_erm = dict()
        if p.cov_rank == 'full':  # backward compat
            if has_rank_arg:
                kwargs['rank'] = 'full'
        else:
            if not has_rank_arg:
                raise RuntimeError(
                    'There is no "rank" argument of compute_covariance, '
                    'you need to update MNE-Python')
            if p.cov_rank is None:
                assert p.compute_rank  # otherwise this is weird
                kwargs['rank'] = _compute_rank(p, subj, run_indices[si])
            else:
                kwargs['rank'] = p.cov_rank
        kwargs_erm['rank'] = kwargs['rank']
        if p.force_erm_cov_rank_full and has_rank_arg:
            kwargs_erm['rank'] = 'full'
        # Use the same thresholds we used for primary Epochs
        if p.autoreject_thresholds:
            reject = get_epochs_evokeds_fnames(p, subj, [])[0][1]
            reject = reject.replace('-epo.fif', '-reject.h5')
            reject = read_hdf5(reject)
        else:
            reject = _handle_dict(p.reject, subj)
        flat = _handle_dict(p.flat, subj)

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
            use_reject, use_flat = _restrict_reject_flat(reject, flat, raw)
            if 'eeg' in use_reject:
                del use_reject['eeg']
            if 'eeg' in use_flat:
                del use_flat['eeg']
            cov = compute_raw_covariance(raw, reject=use_reject, flat=use_flat,
                                         method=p.cov_method, **kwargs_erm)
            write_cov(empty_cov_name, cov)

        # Make evoked covariances
        for ii, (inv_name, inv_run) in enumerate(zip(p.inv_names, p.inv_runs)):
            cov_name = op.join(cov_dir, safe_inserter(inv_name, subj) +
                               ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
            if run_indices[si] is None:
                ridx = inv_run
            else:
                ridx = np.intersect1d(run_indices[si], inv_run)
            # read in raw files
            raw_fnames = get_raw_fnames(p, subj, 'pca', False, False, ridx)
            raw, ratios = _concat_resamp_raws(p, subj, raw_fnames)
            this_decim = _handle_decim(decim[si], raw.info['sfreq'])
            # read in events
            picker = p.pick_events_cov
            if isinstance(picker, str):
                assert picker == 'restrict', \
                    'Only "restrict" is a valid string for p.pick_events_cov'
            elif picker and not callable(picker):
                picker = picker[ii]
            events = _read_events(p, subj, ridx, raw, ratios, picker=picker)
            # handle regex-based reject_epochs_by_annot defs
            reject_epochs_by_annot = _check_reject_annot_regex(p, raw)
            # create epochs
            use_reject, use_flat = _restrict_reject_flat(reject, flat, raw)
            baseline = _get_baseline(p)
            epochs = Epochs(raw, events, event_id=None, tmin=p.bmin,
                            tmax=p.bmax, baseline=baseline,
                            proj=False,
                            reject=use_reject, flat=use_flat, preload=True,
                            decim=this_decim,
                            verbose='error',  # ignore decim-related warnings
                            on_missing=p.on_missing,
                            reject_by_annotation=reject_epochs_by_annot)
            epochs.pick_types(meg=True, eeg=True, exclude=[])
            cov = compute_covariance(epochs, method=p.cov_method,
                                     **kwargs)
            if kwargs.get('rank', None) not in (None, 'full'):
                want_rank = sum(kwargs['rank'].values())
                out_rank = compute_whitener(
                    cov, epochs.info, return_rank=True, verbose='error')[2]
                if want_rank != out_rank:
                    # Hopefully we never hit this code path, but let's keep
                    # some debugging stuff around just in case
                    plot_cov(cov, epochs.info)
                    epochs_fnames, _ = get_epochs_evokeds_fnames(
                        p, subj, p.analyses)
                    epochs2 = read_epochs(epochs_fnames[1], preload=True)
                    idx = np.searchsorted(epochs.events[:, 0],
                                          epochs2.events[:, 0])
                    assert len(np.unique(idx)) == len(idx)
                    epochs = epochs[idx]
                    assert np.array_equal(epochs.events[:, 0],
                                          epochs2.events[:, 0])
                    epochs2.pick_types(meg=True, eeg=True, exclude=[])
                    import matplotlib.pyplot as plt
                    plt.figure()
                    for eps in (epochs, epochs2):
                        eps = eps.get_data().transpose([1, 0, 2])
                        eps = eps.reshape(len(eps), -1)
                        plt.plot(
                            np.log10(np.maximum(linalg.svdvals(eps), 1e-50)))
                    epochs.plot()
                    baseline = _get_baseline(p)
                    epochs2.copy().crop(*baseline).plot()
                    raise RuntimeError('Error computing rank')

            write_cov(cov_name, cov)
        print()
