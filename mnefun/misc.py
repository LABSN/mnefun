# -*- coding: utf-8 -*-
"""Miscellaneous utilities."""

import numpy as np

import mne


def make_montage(info, kind, check=False):
    from . import _reorder
    assert kind in ('mgh60', 'mgh70', 'uw_70', 'uw_60')
    picks = mne.pick_types(info, meg=False, eeg=True, exclude=())
    sphere = mne.make_sphere_model('auto', 'auto', info)
    info = mne.pick_info(info, picks)
    to_names = info['ch_names']
    if kind in ('mgh60', 'mgh70'):
        if kind == 'mgh60':
            assert len(to_names) in (59, 60)
        else:
            assert len(to_names) in (70,)
        montage = mne.channels.make_standard_montage(
            kind, head_size=sphere.radius)
        from_names = mne.utils._clean_names(to_names, remove_whitespace=True)
    else:
        assert len(to_names) == 60
        from_names = getattr(_reorder, 'ch_names_' + kind)
        montage = mne.channels.make_standard_montage(
            'standard_1020', head_size=sphere.radius)
    assert len(from_names) == len(to_names)
    montage_pos = montage._get_ch_pos()
    montage = mne.channels.make_dig_montage(
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
