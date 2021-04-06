# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import warnings

import numpy as np
from mne import find_events, write_events, read_events, concatenate_events
from mne.io import read_raw_fif

from ._paths import get_raw_fnames, get_event_fnames


def default_score(p, subjects, run_indices):
    """Default scoring function that just passes event numbers through"""
    for si, subj in enumerate(subjects):
        print('  Scoring subject %s... ' % subj)

        # Figure out what our filenames should be
        raw_fnames = get_raw_fnames(p, subj, 'raw', False, False,
                                    run_indices[si])
        eve_fnames = get_event_fnames(p, subj, run_indices[si])

        for raw_fname, eve_fname in zip(raw_fnames, eve_fnames):
            with warnings.catch_warnings(record=True):
                raw = read_raw_fif(raw_fname, allow_maxshield='yes')
            events = find_events(raw, stim_channel='STI101', shortest_event=1)
            write_events(eve_fname, events)


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
    presses : list of ndarray
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
    raw = read_raw_fif(fname, allow_maxshield='yes', preload=True)
    raw.pick_types(meg=False, stim=True)
    orig_events = find_events(raw, stim_channel='STI101', shortest_event=0)
    events = list()
    for ch in range(1, 9):
        stim_channel = 'STI%03d' % ch
        ev_101 = find_events(raw, stim_channel='STI101', mask=2 ** (ch - 1),
                             mask_type='and')
        if stim_channel in raw.ch_names:
            ev = find_events(raw, stim_channel=stim_channel)
            if not np.array_equal(ev_101[:, 0], ev[:, 0]):
                warnings.warn('Event coding mismatch between STIM channels')
        else:
            ev = ev_101
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


def _pick_events(events, picker):
    old_cnt = sum(len(e) for e in events)
    events = picker(events)
    new_cnt = len(events)
    print(f'  Using {new_cnt}/{old_cnt} events.')
    return events


def _read_events(p, subj, ridx, raw, picker=None):
    ridx = np.array(ridx)
    assert ridx.ndim == 1
    if picker == 'restrict':  # limit to events that will be processed
        ids = p.in_numbers
        picker = None
        print('    Events restricted to those in params.in_numbers')
    else:
        ids = None
    events = list()
    for fname in get_event_fnames(p, subj, ridx):
        # gracefully handle empty events (e.g., resting state)
        with open(fname, 'r') as fid:
            content = fid.read().strip()
        if not content:
            these_events = np.empty((0, 3), int)
        else:
            these_events = read_events(fname)
            if ids is not None:
                these_events = these_events[np.in1d(these_events[:, 2], ids)]
        events.append(these_events)
    if len(events) == 1 and len(raw._first_samps) > 1:  # for split raw
        first_samps = raw._first_samps[:1]
        last_samps = raw._last_samps[-1:]
    else:
        first_samps = raw._first_samps
        last_samps = raw._last_samps
    events = concatenate_events(events, first_samps, last_samps)
    if picker:
        events = _pick_events(events, picker)
    if len(np.unique(events[:, 0])) != len(events):
        raise RuntimeError('Non-unique event samples found after '
                           'concatenation')
    # do time adjustment
    t_adj = int(np.round(-p.t_adjust * raw.info['sfreq']))
    events[:, 0] += t_adj
    return events
