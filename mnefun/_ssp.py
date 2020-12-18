"""Preprocessing (SSP and filtering)."""

from collections import Counter
import os
import os.path as op
import warnings

import numpy as np
from mne import (concatenate_raws, compute_proj_evoked, compute_proj_epochs,
                 write_proj, pick_types, Epochs, compute_proj_raw, read_proj,
                 make_fixed_length_events, write_events)
from mne.preprocessing import find_ecg_events, find_eog_events
from mne.filter import filter_data
from mne.io import read_raw_fif
from mne.viz import plot_drop_log
from mne.utils import _pl

from ._paths import get_raw_fnames, get_bad_fname
from ._utils import (get_args, _fix_raw_eog_cals, _handle_dict, _safe_remove,
                     _get_baseline, _restrict_reject_flat, _get_epo_kwargs)


def _get_fir_kwargs(fir_design):
    """Get FIR kwargs in backward-compatible way."""
    fir_kwargs = dict()
    old_kwargs = dict()
    if 'fir_design' in get_args(filter_data):
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
               fir_design='firwin2', pick=True,
               skip_by_annotation=('bad', 'skip')):
    """Helper to load, filter, concatenate, then project raw files"""
    from mne.io.proj import _needs_eeg_average_ref_proj
    from ._sss import _read_raw_prebad
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    if disp_files:
        print(f'    Loading and filtering {len(raw_names)} '
              f'file{_pl(raw_names)}.')
    raw = list()
    for ri, rn in enumerate(raw_names):
        if isinstance(bad_file, tuple):
            p, subj, kwargs = bad_file
            r = _read_raw_prebad(p, subj, rn, disp=(ri == 0), **kwargs)
        else:
            r = read_raw_fif(rn, preload=True, allow_maxshield='yes')
            r.load_bad_channels(bad_file, force=force_bads)
        if pick:
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
    _fix_raw_eog_cals(raw)
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


def compute_proj_wrap(epochs, average, **kwargs):
    if average:
        return compute_proj_evoked(epochs.average(), **kwargs)
    else:
        return compute_proj_epochs(epochs, **kwargs)


def _get_pca_dir(p, subj):
    pca_dir = op.join(p.work_dir, subj, p.pca_dir)
    if not op.isdir(pca_dir):
        os.mkdir(pca_dir)
    return pca_dir


def _get_proj_kwargs(p):
    proj_kwargs = dict()
    p_sl = 1
    if 'meg' not in get_args(compute_proj_raw):
        if p.proj_meg != 'separate':
            raise RuntimeError('MNE is too old for proj_meg option')
    else:
        proj_kwargs['meg'] = p.proj_meg
        if p.proj_meg == 'combined':
            p_sl = 2
    return proj_kwargs, p_sl


def _compute_erm_proj(p, subj, projs, kind, bad_file, remove_existing=False,
                      disp_files=None):
    disp_files = p.disp_files if disp_files is None else disp_files
    assert kind in ('sss', 'raw')
    proj_nums = _proj_nums(p, subj)
    proj_kwargs, p_sl = _get_proj_kwargs(p)
    empty_names = get_raw_fnames(p, subj, kind, 'only')
    fir_kwargs, _ = _get_fir_kwargs(p.fir_design)
    flat = _handle_dict(p.flat, subj)
    raw = _raw_LRFCP(
        raw_names=empty_names, sfreq=p.proj_sfreq,
        l_freq=p.erm_proj_hp_cut, h_freq=p.erm_proj_lp_cut,
        n_jobs=p.n_jobs_fir, apply_proj=not remove_existing,
        n_jobs_resample=p.n_jobs_resample, projs=projs,
        bad_file=bad_file, disp_files=disp_files, method='fir',
        filter_length=p.filter_length, force_bads=True,
        l_trans=p.hp_trans, h_trans=p.lp_trans,
        phase=p.phase, fir_window=p.fir_window,
        skip_by_annotation='edge', **fir_kwargs)
    if remove_existing:
        raw.del_proj()
    raw.filter(p.cont_hp, p.cont_lp, n_jobs=p.n_jobs_fir, method='fir',
               filter_length=p.filter_length, h_trans_bandwidth=0.5,
               fir_window=p.fir_window, phase=p.phase,
               skip_by_annotation='edge', **fir_kwargs)
    if projs:
        raw.add_proj(projs)
    raw.apply_proj()
    raw.pick_types(meg=True, eeg=False, exclude=())  # remove EEG
    use_reject = p.erm_proj_reject
    if use_reject is None:
        use_reject = p.reject
    use_reject, use_flat = _restrict_reject_flat(
        _handle_dict(use_reject, subj), flat, raw)
    pr = compute_proj_raw(raw, duration=1, n_grad=proj_nums[2][0],
                          n_mag=proj_nums[2][1], n_eeg=proj_nums[2][2],
                          reject=use_reject, flat=use_flat,
                          n_jobs=p.n_jobs_mkl, **proj_kwargs)
    assert len(pr) == np.sum(proj_nums[2][::p_sl])
    # When doing eSSS it's a bit weird to put this in pca_dir but why not
    pca_dir = _get_pca_dir(p, subj)
    cont_proj = op.join(pca_dir, 'preproc_cont-proj.fif')
    write_proj(cont_proj, pr)
    return pr


def do_preprocessing_combined(p, subjects, run_indices):
    """Do preprocessing on all raw files together.

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
        proj_nums = _proj_nums(p, subj)
        ecg_channel = _handle_dict(p.ecg_channel, subj)
        flat = _handle_dict(p.flat, subj)
        if p.disp_files:
            print('  Preprocessing subject %g/%g (%s).'
                  % (si + 1, len(subjects), subj))
        pca_dir = _get_pca_dir(p, subj)
        bad_file = get_bad_fname(p, subj, check_exists=False)

        # Create SSP projection vectors after marking bad channels
        raw_names = get_raw_fnames(p, subj, 'sss', False, False,
                                   run_indices[si])
        empty_names = get_raw_fnames(p, subj, 'sss', 'only')
        for r in raw_names + empty_names:
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')

        fir_kwargs, old_kwargs = _get_fir_kwargs(p.fir_design)
        if isinstance(p.auto_bad, float):
            print('    Creating post SSS bad channel file:\n'
                  '        %s' % bad_file)
            # do autobad
            raw = _raw_LRFCP(raw_names, p.proj_sfreq, None, None, p.n_jobs_fir,
                             p.n_jobs_resample, list(), None, p.disp_files,
                             method='fir', filter_length=p.filter_length,
                             apply_proj=False, force_bads=False,
                             l_trans=p.hp_trans, h_trans=p.lp_trans,
                             phase=p.phase, fir_window=p.fir_window,
                             pick=True, skip_by_annotation='edge',
                             **fir_kwargs)
            events = fixed_len_events(p, raw)
            rtmin = p.reject_tmin \
                if p.reject_tmin is not None else p.tmin
            rtmax = p.reject_tmax \
                if p.reject_tmax is not None else p.tmax
            # do not mark eog channels bad
            meg, eeg = 'meg' in raw, 'eeg' in raw
            picks = pick_types(raw.info, meg=meg, eeg=eeg, eog=False,
                               exclude=[])
            assert p.auto_bad_flat is None or isinstance(p.auto_bad_flat, dict)
            assert p.auto_bad_reject is None or \
                isinstance(p.auto_bad_reject, dict) or \
                p.auto_bad_reject == 'auto'
            if p.auto_bad_reject == 'auto':
                print('    Auto bad channel selection active. '
                      'Will try using Autoreject module to '
                      'compute rejection criterion.')
                try:
                    from autoreject import get_rejection_threshold
                except ImportError:
                    raise ImportError('     Autoreject module not installed.\n'
                                      '     Noisy channel detection parameter '
                                      '     not defined. To use autobad '
                                      '     channel selection either define '
                                      '     rejection criteria or install '
                                      '     Autoreject module.\n')
                print('    Computing thresholds.\n', end='')
                temp_epochs = Epochs(
                    raw, events, event_id=None, tmin=rtmin, tmax=rtmax,
                    baseline=_get_baseline(p), proj=True, reject=None,
                    flat=None, preload=True, decim=1)
                kwargs = dict()
                if 'verbose' in get_args(get_rejection_threshold):
                    kwargs['verbose'] = False
                reject = get_rejection_threshold(temp_epochs, **kwargs)
                reject = {kk: vv for kk, vv in reject.items()}
            elif p.auto_bad_reject is None and p.auto_bad_flat is None:
                raise RuntimeError('Auto bad channel detection active. Noisy '
                                   'and flat channel detection '
                                   'parameters not defined. '
                                   'At least one criterion must be defined.')
            else:
                reject = p.auto_bad_reject
            if 'eog' in reject.keys():
                reject.pop('eog', None)
            epochs = Epochs(raw, events, None, tmin=rtmin, tmax=rtmax,
                            baseline=_get_baseline(p), picks=picks,
                            reject=reject, flat=p.auto_bad_flat,
                            proj=True, preload=True, decim=1,
                            reject_tmin=rtmin, reject_tmax=rtmax)
            # channel scores from drop log
            drops = Counter([ch for d in epochs.drop_log for ch in d])
            # get rid of non-channel reasons in drop log
            scores = {kk: vv for kk, vv in drops.items() if
                      kk in epochs.ch_names}
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
            print('    Clearing bad channels (no file %s)'
                  % op.sep.join(bad_file.split(op.sep)[-3:]))
            bad_file = None

        ecg_t_lims = p.ecg_t_lims
        ecg_f_lims = p.ecg_f_lims

        ecg_eve = op.join(pca_dir, 'preproc_ecg-eve.fif')
        ecg_epo = op.join(pca_dir, 'preproc_ecg-epo.fif')
        ecg_proj = op.join(pca_dir, 'preproc_ecg-proj.fif')
        all_proj = op.join(pca_dir, 'preproc_all-proj.fif')

        get_projs_from = _handle_dict(p.get_projs_from, subj)
        if get_projs_from is None:
            get_projs_from = np.arange(len(raw_names))
        pre_list = [r for ri, r in enumerate(raw_names)
                    if ri in get_projs_from]

        projs = list()
        raw_orig = _raw_LRFCP(
            raw_names=pre_list, sfreq=p.proj_sfreq, l_freq=None, h_freq=None,
            n_jobs=p.n_jobs_fir, n_jobs_resample=p.n_jobs_resample,
            projs=projs, bad_file=bad_file, disp_files=p.disp_files,
            method='fir', filter_length=p.filter_length, force_bads=False,
            l_trans=p.hp_trans, h_trans=p.lp_trans, phase=p.phase,
            fir_window=p.fir_window, pick=True, skip_by_annotation='edge',
            **fir_kwargs)

        # Apply any user-supplied extra projectors
        if p.proj_extra is not None:
            if p.disp_files:
                print('    Adding extra projectors from "%s".' % p.proj_extra)
            projs.extend(read_proj(op.join(pca_dir, p.proj_extra)))

        proj_kwargs, p_sl = _get_proj_kwargs(p)
        #
        # Calculate and apply ERM projectors
        #
        if not p.erm_proj_as_esss:
            if any(proj_nums[2]):
                assert proj_nums[2][2] == 0  # no EEG projectors for ERM
                if len(empty_names) == 0:
                    raise RuntimeError('Cannot compute empty-room projectors '
                                    'from continuous raw data')
                if p.disp_files:
                    print('    Computing continuous projectors using ERM.')
                # Use empty room(s), but processed the same way
                projs.extend(_compute_erm_proj(p, subj, 'sss', projs, bad_file))
            else:
                cont_proj = op.join(pca_dir, 'preproc_cont-proj.fif')
                _safe_remove(cont_proj)

        #
        # Calculate and apply the ECG projectors
        #
        if any(proj_nums[0]):
            if p.disp_files:
                print('    Computing ECG projectors...', end='')
            raw = raw_orig.copy()

            raw.filter(ecg_f_lims[0], ecg_f_lims[1], n_jobs=p.n_jobs_fir,
                       method='fir', filter_length=p.filter_length,
                       l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                       phase='zero-double', fir_window='hann',
                       skip_by_annotation='edge', **old_kwargs)
            raw.add_proj(projs)
            raw.apply_proj()
            find_kwargs = dict()
            if 'reject_by_annotation' in get_args(find_ecg_events):
                find_kwargs['reject_by_annotation'] = True
            elif len(raw.annotations) > 0:
                print('    WARNING: ECG event detection will not make use of '
                      'annotations, please update MNE-Python')
            # We've already filtered the data channels above, but this
            # filters the ECG channel
            ecg_events = find_ecg_events(
                raw, 999, ecg_channel, 0., ecg_f_lims[0], ecg_f_lims[1],
                qrs_threshold='auto', return_ecg=False, **find_kwargs)[0]
            use_reject, use_flat = _restrict_reject_flat(
                _handle_dict(p.ssp_ecg_reject, subj), flat, raw)
            ecg_epochs = Epochs(
                raw, ecg_events, 999, ecg_t_lims[0], ecg_t_lims[1],
                baseline=None, reject=use_reject, flat=use_flat, preload=True)
            print('  obtained %d epochs from %d events.' % (len(ecg_epochs),
                                                            len(ecg_events)))
            if len(ecg_epochs) >= 20:
                write_events(ecg_eve, ecg_epochs.events)
                ecg_epochs.save(ecg_epo, **_get_epo_kwargs())
                desc_prefix = 'ECG-%s-%s' % tuple(ecg_t_lims)
                pr = compute_proj_wrap(
                    ecg_epochs, p.proj_ave, n_grad=proj_nums[0][0],
                    n_mag=proj_nums[0][1], n_eeg=proj_nums[0][2],
                    desc_prefix=desc_prefix, **proj_kwargs)
                assert len(pr) == np.sum(proj_nums[0][::p_sl])
                write_proj(ecg_proj, pr)
                projs.extend(pr)
            else:
                plot_drop_log(ecg_epochs.drop_log)
                raw.plot(events=ecg_epochs.events)
                raise RuntimeError('Only %d/%d good ECG epochs found'
                                   % (len(ecg_epochs), len(ecg_events)))
            del raw, ecg_epochs, ecg_events
        else:
            _safe_remove([ecg_proj, ecg_eve, ecg_epo])

        #
        # Next calculate and apply the EOG projectors
        #
        for idx, kind in ((1, 'EOG'), (3, 'HEOG'), (4, 'VEOG')):
            _compute_add_eog(
                p, subj, raw_orig, projs, proj_nums[idx], kind, pca_dir,
                flat, proj_kwargs, old_kwargs, p_sl)
        del proj_nums

        # save the projectors
        write_proj(all_proj, projs)

        #
        # Look at raw_orig for trial DQs now, it will be quick
        #
        raw_orig.filter(p.hp_cut, p.lp_cut, n_jobs=p.n_jobs_fir, method='fir',
                        filter_length=p.filter_length,
                        l_trans_bandwidth=p.hp_trans, phase=p.phase,
                        h_trans_bandwidth=p.lp_trans, fir_window=p.fir_window,
                        skip_by_annotation='edge', **fir_kwargs)
        raw_orig.add_proj(projs)
        raw_orig.apply_proj()
        # now let's epoch with 1-sec windows to look for DQs
        events = fixed_len_events(p, raw_orig)
        reject = _handle_dict(p.reject, subj)
        use_reject, use_flat = _restrict_reject_flat(reject, flat, raw_orig)
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


def _proj_nums(p, subj):
    proj_nums = np.array(_handle_dict(p.proj_nums, subj), int)
    if proj_nums.shape not in ((3, 3), (4, 3), (5, 3)):
        raise ValueError('proj_nums for %s must be an array with shape '
                         '(3, 3), (4, 3), or (5, 3), got %s'
                         % (subj, proj_nums.shape))
    proj_nums = np.pad(
        proj_nums, ((0, 5 - proj_nums.shape[0]), (0, 0)), 'constant')
    assert proj_nums.shape == (5, 3)
    return proj_nums


def _compute_add_eog(p, subj, raw_orig, projs, eog_nums, kind, pca_dir,
                     flat, proj_kwargs, old_kwargs, p_sl):
    assert kind in ('EOG', 'HEOG', 'VEOG')
    bk = dict(EOG='blink').get(kind, kind.lower())
    eog_eve = op.join(pca_dir, f'preproc_{bk}-eve.fif')
    eog_epo = op.join(pca_dir, f'preproc_{bk}-epo.fif')
    eog_proj = op.join(pca_dir, f'preproc_{bk}-proj.fif')
    eog_t_lims = _handle_dict(getattr(p, f'{kind.lower()}_t_lims'), subj)
    eog_f_lims = _handle_dict(getattr(p, f'{kind.lower()}_f_lims'), subj)
    eog_channel = _handle_dict(getattr(p, f'{kind.lower()}_channel'), subj)
    thresh = _handle_dict(getattr(p, f'{kind.lower()}_thresh'), subj)
    if eog_channel is None and kind != 'EOG':
        eog_channel = 'EOG061' if kind == 'HEOG' else 'EOG062'
    if eog_nums.any():
        if p.disp_files:
            print(f'    Computing {kind} projectors...', end='')
        raw = raw_orig.copy()
        raw.filter(eog_f_lims[0], eog_f_lims[1], n_jobs=p.n_jobs_fir,
                   method='fir', filter_length=p.filter_length,
                   l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                   phase='zero-double', fir_window='hann',
                   skip_by_annotation='edge', **old_kwargs)
        raw.add_proj(projs)
        raw.apply_proj()
        eog_events = find_eog_events(
            raw, ch_name=eog_channel, reject_by_annotation=True,
            thresh=thresh)
        use_reject, use_flat = _restrict_reject_flat(
            _handle_dict(p.ssp_eog_reject, subj), flat, raw)
        eog_epochs = Epochs(
            raw, eog_events, 998, eog_t_lims[0], eog_t_lims[1],
            baseline=None, reject=use_reject, flat=use_flat, preload=True)
        print('  obtained %d epochs from %d events.' % (len(eog_epochs),
                                                        len(eog_events)))
        del eog_events
        if len(eog_epochs) >= 5:
            write_events(eog_eve, eog_epochs.events)
            eog_epochs.save(eog_epo, **_get_epo_kwargs())
            desc_prefix = f'{kind}-%s-%s' % tuple(eog_t_lims)
            pr = compute_proj_wrap(
                eog_epochs, p.proj_ave, n_grad=eog_nums[0],
                n_mag=eog_nums[1], n_eeg=eog_nums[2],
                desc_prefix=desc_prefix, **proj_kwargs)
            assert len(pr) == np.sum(eog_nums[::p_sl])
            write_proj(eog_proj, pr)
            projs.extend(pr)
        else:
            warnings.warn('Only %d usable EOG events!' % len(eog_epochs))
            _safe_remove([eog_proj, eog_eve, eog_epo])
        del raw, eog_epochs
    else:
        _safe_remove([eog_proj, eog_eve, eog_epo])


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
        bad_file = get_bad_fname(p, subj)
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
            from ._viz import _viz_raw_ssp_events
            _viz_raw_ssp_events(p, subj, run_indices[si])


class FakeEpochs(object):
    """Make iterable epoch-like class, convenient for MATLAB transition"""

    def __init__(self, data, ch_names, tmin=-0.2, sfreq=1000.0):
        raise RuntimeError('Use mne.EpochsArray instead')


def fixed_len_events(p, raw):
    """Create fixed length trial events from raw object"""
    dur = p.tmax - p.tmin
    events = make_fixed_length_events(raw, 1, duration=dur)
    return events
