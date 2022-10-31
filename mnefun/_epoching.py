import os
import os.path as op
import warnings

from h5io import write_hdf5
import numpy as np
from scipy import io as spio
from mne import Epochs, write_evokeds
from mne.defaults import DEFAULTS
from mne.epochs import combine_event_ids
from mne.io import read_raw_fif, concatenate_raws
from mne.viz import plot_drop_log
from mne.utils import use_log_level

from ._paths import get_raw_fnames, get_epochs_evokeds_fnames
from ._scoring import _read_events
from ._sss import _read_raw_prebad
from ._utils import (_fix_raw_eog_cals, _get_baseline, get_args, _handle_dict,
                     _restrict_reject_flat, _get_epo_kwargs, _handle_decim,
                     _check_reject_annot_regex)


def save_epochs(p, subjects, in_names, in_numbers, analyses, out_names,
                out_numbers, must_match, decim, run_indices):
    """Generate epochs from raw data based on events.

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
        raw, ratios = _concat_resamp_raws(p, subj, raw_names)
        assert ratios.shape == (len(raw_names),)
        # optionally calculate autoreject thresholds
        this_decim = _handle_decim(decim[si], raw.info['sfreq'])
        new_sfreq = raw.info['sfreq'] / this_decim
        if p.disp_files:
            print('    Epoching data (decim=%s -> sfreq=%0.1f Hz).'
                  % (this_decim, new_sfreq))
        if new_sfreq not in sfreqs:
            if len(sfreqs) > 0:
                warnings.warn('resulting new sampling frequency %s not equal '
                              'to previous values %s' % (new_sfreq, sfreqs))
            sfreqs.add(new_sfreq)
        epochs_fnames, evoked_fnames = get_epochs_evokeds_fnames(p, subj,
                                                                 analyses)
        mat_file, fif_file = epochs_fnames
        # handle regex-based reject_epochs_by_annot defs
        reject_epochs_by_annot = _check_reject_annot_regex(p, raw)
        if p.autoreject_thresholds:
            assert len(p.autoreject_types) > 0
            assert all(a in ('mag', 'grad', 'eeg', 'ecg', 'eog')
                       for a in p.autoreject_types)
            from autoreject import get_rejection_threshold
            picker = p.pick_events_autoreject
            if type(picker) is str:
                assert picker == 'restrict', \
                    'Only "restrict" is valid str for p.pick_events_autoreject'
            events = _read_events(
                p, subj, run_indices[si], raw, ratios, picker=picker)
            print('    Computing autoreject thresholds', end='')
            rtmin = p.reject_tmin if p.reject_tmin is not None else p.tmin
            rtmax = p.reject_tmax if p.reject_tmax is not None else p.tmax
            temp_epochs = Epochs(
                raw, events, event_id=None, tmin=rtmin, tmax=rtmax,
                baseline=_get_baseline(p), proj=True, reject=None,
                flat=None, preload=True, decim=this_decim,
                reject_by_annotation=reject_epochs_by_annot)
            kwargs = dict()
            if 'verbose' in get_args(get_rejection_threshold):
                kwargs['verbose'] = False
            new_dict = get_rejection_threshold(temp_epochs, **kwargs)
            use_reject = dict()
            msgs = list()
            for k in p.autoreject_types:
                msgs.append('%s=%d %s'
                            % (k, DEFAULTS['scalings'][k] * new_dict[k],
                               DEFAULTS['units'][k]))
                use_reject[k] = new_dict[k]
            print(': ' + ', '.join(msgs))
            hdf5_file = fif_file.replace('-epo.fif', '-reject.h5')
            assert hdf5_file.endswith('.h5')
            write_hdf5(hdf5_file, use_reject, overwrite=True)
        else:
            use_reject = _handle_dict(p.reject, subj)

        # read in events and create epochs
        events = _read_events(p, subj, run_indices[si], raw, ratios,
                              picker='restrict')
        if len(events) == 0:
            raise ValueError('No valid events found')
        flat = _handle_dict(p.flat, subj)
        use_reject, use_flat = _restrict_reject_flat(use_reject, flat, raw)
        epochs = Epochs(raw, events, event_id=old_dict, tmin=p.tmin,
                        tmax=p.tmax, baseline=_get_baseline(p),
                        reject=use_reject, flat=use_flat, proj=p.epochs_proj,
                        preload=True, decim=this_decim,
                        on_missing=p.on_missing,
                        reject_tmin=p.reject_tmin, reject_tmax=p.reject_tmax,
                        reject_by_annotation=reject_epochs_by_annot)
        if epochs.events.shape[0] < 1:
            _raise_bad_epochs(raw, epochs, events, plot=p.plot_drop_logs)
        del raw
        drop_logs.append(epochs.drop_log)
        ch_namess.append(epochs.ch_names)
        # only kept trials that were not dropped
        sfreq = epochs.info['sfreq']
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
            in_names_match = list(in_names[match])
            # use some variables to allow safe name re-use
            offset = max(epochs.events[:, 2].max(), new_numbers.max()) + 1
            safety_str = '__mnefun_copy__'
            assert len(new_numbers) == len(names)  # checked above
            if p.match_fun is None:
                e = None
            else:  # use custom matching
                args = [epochs.copy(), analysis, nn, in_names_match, names]
                if len(get_args(p.match_fun)) > 5:
                    args = args + [numbers]
                e = p.match_fun(*args)
            if e is None:
                # first, equalize trial counts (this will make a copy)
                e = epochs[list(in_names[numbers > 0])]
                # some could be missing
                in_names_match = [
                    name for name in in_names_match if name in e.event_id]
                if len(in_names_match) > 1:
                    print(f'      Equalizing: {in_names_match}')
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

            # now make evoked for each out type
            evokeds = list()
            n_standard = 0
            kinds = ['standard']
            if p.every_other:
                kinds += ['even', 'odd']
            for kind in kinds:
                for name in names:
                    this_e = e[name]
                    if kind == 'even':
                        this_e = this_e[::2]
                    elif kind == 'odd':
                        this_e = this_e[1::2]
                    else:
                        assert kind == 'standard'
                    with use_log_level('error'):
                        with warnings.catch_warnings(record=True):
                            warnings.simplefilter('ignore')
                            ave = this_e.average(picks='all')
                            ave.comment = name
                            stde = this_e.standard_error(picks='all')
                            stde.comment = name
                    if kind != 'standard':
                        ave.comment += ' %s' % (kind,)
                        stde.comment += ' %s' % (kind,)
                    evokeds.append(ave)
                    evokeds.append(stde)
                    if kind == 'standard':
                        n_standard += 2
            write_evokeds(fn, evokeds)
            naves = [str(n) for n in sorted(set([
                evoked.nave for evoked in evokeds[:n_standard]]))]
            bad = [evoked.comment for evoked in evokeds[:n_standard:2]
                   if evoked.nave == 0]
            if bad:
                print(f'      Got 0 epochs for: {bad}')
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
            epochs.save(fif_file, **_get_epo_kwargs())

    if p.plot_drop_logs:
        for subj, drop_log in zip(subjects, drop_logs):
            plot_drop_log(drop_log, threshold=p.drop_thresh, subject=subj)


def _concat_resamp_raws(p, subj, fnames, fix='EOG', prebad=False,
                        preload=None, set_dev_head_t=False):
    raws = []
    first_samps = []
    last_samps = []
    fixed_dht = 0
    for ri, raw_fname in enumerate(fnames):
        if prebad:
            raw = _read_raw_prebad(p, subj, raw_fname, False)
        else:
            raw = read_raw_fif(
                raw_fname, preload=False, allow_maxshield='yes')
        raws.append(raw)
        first_samps.append(raw._first_samps[0])
        last_samps.append(raw._last_samps[-1])
        fixed_dht = raw.info['dev_head_t'] if ri == 0 else fixed_dht
        if set_dev_head_t:
            raw.info['dev_head_t'] = fixed_dht
        del raw
    assert len(raws) > 0
    rates = np.array([r.info['sfreq'] for r in raws], float)
    ratios = rates[0] / rates
    assert rates.shape == (len(fnames),)
    if not (ratios == 1).all():
        if not p.allow_resample:
            raise RuntimeError(
                'Raw sample rates do not match, consider using '
                f'params.allow_resample=True:\n{rates}')
        for ri, (raw, ratio) in enumerate(zip(raws[1:], ratios[1:])):
            if ratio != 1:
                fr, to = raws[0].info['sfreq'], raw.info['sfreq']
                print(f'    Resampling raw {ri + 1}/{len(raws)} ({fr}→{to})')
                raw.load_data().resample(raws[0].info['sfreq'])
    _fix_raw_eog_cals(raws, fix)  # safe b/c cov only needs MEEG
    assert len(ratios) == len(fnames)
    bads = raws[0].info['bads']
    if prebad:
        bads = sorted(set(sum((r.info['bads'] for r in raws), [])))
        for r in raws:
            r.info['bads'] = bads
    raw = concatenate_raws(raws, preload=preload)
    assert raw.info['bads'] == bads
    return raw, ratios


def _raise_bad_epochs(raw, epochs, events, kind=None, plot=True):
    extra = '' if kind is None else f' of type {kind} '
    if plot:
        plot_drop_log(epochs.drop_log)
        raw.plot(events=events)
    raise RuntimeError(
        f'Only {len(epochs)}/{len(events)} good epochs found{extra}')
