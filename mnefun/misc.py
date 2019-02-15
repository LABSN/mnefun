# -*- coding: utf-8 -*-
"""Miscellaneous utilities."""

import numpy as np

import mne


def make_montage(info, kind, check=False):
    from . import _reorder
    assert kind in ('mgh60', 'mgh70', 'uw_70', 'uw_60')
    picks = mne.pick_types(info, meg=False, eeg=True, exclude=())
    if kind in ('mgh60', 'mgh70'):
        ch_names = mne.utils._clean_names(
            [info['ch_names'][pick] for pick in picks], remove_whitespace=True)
        if kind == 'mgh60':
            assert len(ch_names) in (59, 60)
        else:
            assert len(ch_names) in (70,)
        montage = mne.channels.read_montage(kind, ch_names=ch_names)
    else:
        ch_names = getattr(_reorder, 'ch_names_' + kind)
        ch_names = ch_names
        montage = mne.channels.read_montage('standard_1020', ch_names=ch_names)
        assert len(montage.ch_names) == len(ch_names)
        montage.ch_names = ['EEG%03d' % ii for ii in range(1, 61)]
    sphere = mne.make_sphere_model('auto', 'auto', info)
    montage.pos /= np.linalg.norm(montage.pos, axis=-1, keepdims=True)
    montage.pos *= sphere.radius
    montage.pos += sphere['r0']
    info = mne.pick_info(info, picks)
    eeg_pos = np.array([ch['loc'][:3] for ch in info['chs']])
    assert len(eeg_pos) == len(montage.pos), (len(eeg_pos), len(montage.pos))
    if check:
        from mayavi import mlab
        mlab.figure(size=(800, 800))
        mlab.points3d(*sphere['r0'], scale_factor=2 * sphere.radius,
                      color=(0., 0., 1.), opacity=0.1, mode='sphere')
        mlab.points3d(*montage.pos.T, scale_factor=0.01,
                      color=(1, 0, 0), mode='sphere', opacity=0.5)
        mlab.points3d(*eeg_pos.T, scale_factor=0.005, color=(1, 1, 1),
                      mode='sphere', opacity=1)
    return montage, sphere
