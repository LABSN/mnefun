"""Miscellaneous utilities."""

from functools import reduce
import os
import os.path as op
import subprocess
import warnings

import numpy as np
from mne import pick_types, pick_info, make_sphere_model, DipoleFixed, Epochs
from mne.channels import make_standard_montage, make_dig_montage
from mne.fixes import _get_args as get_args  # noqa: F401


def _fix_raw_eog_cals(raws, kind='EOG'):
    """Fix for annoying issue where EOG cals don't match."""
    # Warning: this will only produce correct EOG scalings with preloaded
    # raw data!
    if kind == 'EOG':
        picks = pick_types(raws[0].info, eeg=False, meg=False, eog=True,
                           exclude=[])
    else:
        assert kind == 'all'
        picks = np.arange(len(raws[0].ch_names))
    if len(picks) > 0:
        first_cals = _cals(raws[0])[picks]
        for ri, r in enumerate(raws[1:]):
            if kind == 'EOG':
                picks_2 = pick_types(r.info, eeg=False, meg=False, eog=True,
                                     exclude=[])
            else:
                picks_2 = np.arange(len(r.ch_names))

            assert np.array_equal(picks, picks_2)
            these_cals = _cals(r)[picks]
            if not np.array_equal(first_cals, these_cals):
                warnings.warn('Adjusting %s cals for %s'
                              % (kind, op.basename(r._filenames[0])))
                _cals(r)[picks] = first_cals


def _cals(raw):
    """Helper to deal with the .cals->._cals attribute change."""
    try:
        return raw._cals
    except AttributeError:
        return raw.cals


def _get_baseline(p):
    """Helper to extract baseline from params."""
    if p.baseline == 'individual':
        baseline = (p.bmin, p.bmax)
    else:
        baseline = p.baseline
    # XXX this and some downstream stuff (e.g., tmin=-baseline[0]) won't work
    # for baseline=None, but we can fix that when someone needs it
    baseline = tuple(baseline)
    if baseline[0] is None:
        baseline = (p.tmin, baseline[1])
    if baseline[1] is None:
        baseline = (baseline[0], p.tmax)
    return baseline


def _handle_dict(entry, subj):
    out = entry
    if isinstance(entry, dict):
        try:
            out = entry[subj]
        except KeyError:
            pass
    return out


def _safe_remove(fnames):
    if isinstance(fnames, str):
        fnames = [fnames]
    for fname in fnames:
        if op.isfile(fname):
            os.remove(fname)


def _restrict_reject_flat(reject, flat, raw):
    """Restrict a reject and flat dict based on channel presence"""
    reject = {} if reject is None else reject
    flat = {} if flat is None else flat
    assert isinstance(reject, dict)
    assert isinstance(flat, dict)
    use_reject, use_flat = dict(), dict()
    for in_, out in zip([reject, flat], [use_reject, use_flat]):
        use_keys = [key for key in in_.keys() if key in raw]
        for key in use_keys:
            out[key] = in_[key]
    return use_reject, use_flat


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


def make_montage(info, kind, check=False):
    from mne.utils import _clean_names
    from . import _reorder
    assert kind in ('mgh60', 'mgh70', 'uw_70', 'uw_60')
    picks = pick_types(info, meg=False, eeg=True, exclude=())
    sphere = make_sphere_model('auto', 'auto', info)
    info = pick_info(info, picks)
    to_names = info['ch_names']
    if kind in ('mgh60', 'mgh70'):
        if kind == 'mgh60':
            assert len(to_names) in (59, 60)
        else:
            assert len(to_names) in (70,)
        montage = make_standard_montage(
            kind, head_size=sphere.radius)
        from_names = _clean_names(to_names, remove_whitespace=True)
    else:
        assert len(to_names) == 60
        from_names = getattr(_reorder, 'ch_names_' + kind)
        montage = make_standard_montage(
            'standard_1020', head_size=sphere.radius)
    assert len(from_names) == len(to_names)
    montage_pos = montage._get_ch_pos()
    montage = make_dig_montage(
        {to: montage_pos[fro] for fro, to in zip(from_names, to_names)},
        coord_frame='head')
    eeg_pos = np.array([ch['loc'][:3] for ch in info['chs']])
    montage_pos = montage._get_ch_pos()
    montage_pos = np.array([montage_pos[name] for name in to_names])
    assert len(eeg_pos) == len(montage_pos)
    if check:
        from mayavi import mlab
        mlab.figure(size=(800, 800))
        mlab.points3d(*sphere['r0'], scale_factor=2 * sphere.radius,
                      color=(0., 0., 1.), opacity=0.1, mode='sphere')
        mlab.points3d(*montage_pos.T, scale_factor=0.01,
                      color=(1, 0, 0), mode='sphere', opacity=0.5)
        mlab.points3d(*eeg_pos.T, scale_factor=0.005, color=(1, 1, 1),
                      mode='sphere', opacity=1)
    return montage, sphere


def compute_auc(dip, tmin=-np.inf, tmax=np.inf):
    """Compute the AUC values for a DipoleFixed object."""
    from mne.utils import _time_mask
    if not isinstance(dip, DipoleFixed):
        raise TypeError('dip must be a DipoleFixed, got "%s"' % (type(dip),))
    pick = pick_types(dip.info, meg=False, dipole=True)
    if len(pick) != 1:
        raise RuntimeError('Could not find dipole data')
    time_mask = _time_mask(dip.times, tmin, tmax, dip.info['sfreq'])
    data = dip.data[pick[0], time_mask]
    return np.sum(np.abs(data)) * len(data) * (1. / dip.info['sfreq'])


def _get_epo_kwargs():
    from mne.fixes import _get_args
    epo_kwargs = dict(verbose=False)
    if 'overwrite' in _get_args(Epochs.save):
        epo_kwargs['overwrite'] = True
    return epo_kwargs
