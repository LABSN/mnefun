#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate head movements and neural activations

You can do for example:

$ simulate_movement.py --raw test_raw.fif
                       --pos test_raw_hp.txt
                       --dipoles dips.txt
                       --cov simple
                       --out test_sim_raw.fif
                       --jobs 2 --overwrite --plot

At a minimum --raw, --dipoles, and --pos must be specified. For
simplicity, a spherical head model will be constructed and used.
"""

from __future__ import print_function

import sys
import time
import warnings
import os.path as op
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import mne
from mne.io.constants import FIFF
from mne.commands.utils import get_optparser
from mnefun import simulate_movement


class printer(object):
    def __init__(self, string, section=False):
        start = '\n' if section else ''
        self.str = start + string + ' '
        self.section = section

    def __enter__(self):
        self.t0 = time.time()
        end = '\n' if self.section else ' '
        print(self.str.ljust(50, '.'), end=end)
        sys.stdout.flush()

    def __exit__(self, type_, value, traceback):
        if value is None:
            end = '\n' if self.section else ''
            print('done (%0.1f sec)%s' % (time.time() - self.t0, end))
            sys.stdout.flush()


def run():
    t0 = time.time()
    parser = get_optparser(__file__)
    parser.add_option("--raw", dest="raw_in",
                      help="Input raw FIF file", metavar="FILE")
    parser.add_option("--pos", dest="pos", default=None,
                      help="Position definition text file. Can be 'constant' "
                      "to hold the head position fixed", metavar="FILE")
    parser.add_option("--dipoles", dest="dipoles", default=None,
                      help="Dipole definition file", metavar="FILE")
    parser.add_option("--cov", dest="cov",
                      help="Covariance to use for noise generation. Can be "
                      "'simple' to use a diagonal covariance, or 'off' to "
                      "omit noise",
                      metavar="FILE", default='simple')
    parser.add_option("--duration", dest="duration", default=None,
                      help="Duration of each epoch (sec). If omitted, the last"
                      " time point in the dipole definition file plus 200 ms "
                      "will be used", type="float")
    parser.add_option("-j", "--jobs", dest="n_jobs", help="Number of jobs to"
                      " run in parallel", type="int", default=1)
    parser.add_option("--out", dest="raw_out",
                      help="Output raw filename", metavar="FILE")
    parser.add_option("--plot-dipoles", dest="plot_dipoles", help="Plot "
                      "input dipole positions", action="store_true")
    parser.add_option("--plot-raw", dest="plot_raw", help="Plot the resulting "
                      "raw traces", action="store_true")
    parser.add_option("--plot-evoked", dest="plot_evoked", help="Plot evoked "
                      "data", action="store_true")
    parser.add_option("-p", "--plot", dest="plot", help="Plot dipoles, raw, "
                      "and evoked", action="store_true")
    parser.add_option("--overwrite", dest="overwrite", help="Overwrite the"
                      "output file if it exists", action="store_true")
    options, args = parser.parse_args()

    raw_in = options.raw_in
    pos = options.pos
    raw_out = options.raw_out
    dipoles = options.dipoles
    n_jobs = options.n_jobs
    plot = options.plot
    plot_dipoles = options.plot_dipoles or plot
    plot_raw = options.plot_raw or plot
    plot_evoked = options.plot_evoked or plot
    overwrite = options.overwrite
    duration = options.duration
    cov = options.cov

    # check parameters
    if not (raw_out or plot_raw or plot_evoked):
        raise ValueError('data must either be saved (--out) or '
                         'plotted (--plot-raw or --plot_evoked)')
    if raw_out and op.isfile(raw_out) and not overwrite:
        raise ValueError('output file exists, use --overwrite (%s)' % raw_out)

    if raw_in is None or pos is None or dipoles is None:
        parser.print_help()
        sys.exit(1)

    s = 'Simulate raw data with head movements'
    print('\n%s\n%s\n%s\n' % ('-' * len(s), s, '-' * len(s)))

    # setup the simulation

    with printer('Reading dipole definitions'):
        if not op.isfile(dipoles):
            raise IOError('dipole file not found:\n%s' % dipoles)
        dipoles = np.loadtxt(dipoles, skiprows=1, dtype=float)
        n_dipoles = dipoles.shape[0]
        if dipoles.shape[1] != 8:
            raise ValueError('dipoles must have 8 columns')
        rr = dipoles[:, :3] * 1e-3
        nn = dipoles[:, 3:6]
        t = dipoles[:, 6:8]
        duration = t.max() + 0.2 if duration is None else duration
        if (t[:, 0] > t[:, 1]).any():
            raise ValueError('found tmin > tmax in dipole file')
        if (t < 0).any():
            raise ValueError('found t < 0 in dipole file')
        if (t > duration).any():
            raise ValueError('found t > duration in dipole file')
        amp = np.sqrt(np.sum(nn * nn, axis=1)) * 1e-9
        mne.surface._normalize_vectors(nn)
        nn[(nn == 0).all(axis=1)] = (1, 0, 0)
        src = mne.SourceSpaces([
            dict(rr=rr, nn=nn, inuse=np.ones(n_dipoles, int),
                 coord_frame=FIFF.FIFFV_COORD_HEAD)])
        for key in ['pinfo', 'nuse_tri', 'use_tris', 'patch_inds']:
            src[0][key] = None
        trans = {'from': FIFF.FIFFV_COORD_HEAD, 'to': FIFF.FIFFV_COORD_MRI,
                 'trans': np.eye(4)}
        if (amp > 100e-9).any():
            print('')
            warnings.warn('Largest dipole amplitude %0.1f > 100 nA'
                          % (amp.max() * 1e9))

    if pos == 'constant':
        print('Holding head position constant')
        pos = None
    else:
        with printer('Loading head positions'):
            pos = mne.get_chpi_positions(pos)

    with printer('Loading raw data file'):
        with warnings.catch_warnings(record=True):
            raw = mne.io.Raw(raw_in, preload=False, allow_maxshield=True,
                             verbose=False)

    if cov == 'simple':
        print('Using diagonal covariance for brain noise')
    elif cov == 'off':
        print('Omitting brain noise in the simulation')
        cov = None
    else:
        with printer('Loading covariance file for brain noise'):
            cov = mne.read_cov(cov)

    with printer('Setting up spherical model'):
        bem = mne.bem.make_sphere_model('auto', 'auto', raw.info,
                                        verbose=False)
        # check that our sources are reasonable
        rad = bem['layers'][0]['rad']
        r0 = bem['r0']
        outside = np.sqrt(np.sum((rr - r0) ** 2, axis=1)) >= rad
        n_outside = outside.sum()
        if n_outside > 0:
            print('')
            raise ValueError(
                '%s dipole%s outside the spherical model, are your positions '
                'in mm?' % (n_outside, 's were' if n_outside != 1 else ' was'))

    with printer('Constructing source estimate'):
        tmids = t.mean(axis=1)
        t = np.round(t * raw.info['sfreq']).astype(int)
        t[:, 1] += 1  # make it inclusive
        n_samp = int(np.ceil(duration * raw.info['sfreq']))
        data = np.zeros((n_dipoles, n_samp))
        for di, (t_, amp_) in enumerate(zip(t, amp)):
            data[di, t_[0]:t_[1]] = amp_ * np.hanning(t_[1] - t_[0])
        stc = mne.VolSourceEstimate(data, np.arange(n_dipoles),
                                    0, 1. / raw.info['sfreq'])

    # do the simulation
    print('')
    raw_mv = simulate_movement(raw, pos, stc, trans, src, bem, cov,
                               n_jobs=n_jobs, verbose=True)
    print('')

    if raw_out:
        with printer('Saving data'):
            raw_mv.save(raw_out, overwrite=overwrite)

    # plot results -- must be *after* save because we low-pass filter
    if plot_dipoles:
        with printer('Plotting dipoles'):
            fig, axs = plt.subplots(1, 3, figsize=(10, 3), facecolor='w')
            fig.canvas.set_window_title('Dipoles')
            meg_info = mne.pick_info(raw.info,
                                     mne.pick_types(raw.info,
                                                    meg=True, eeg=False))
            helmet_rr = [ch['coil_trans'][:3, 3].copy()
                         for ch in meg_info['chs']]
            helmet_nn = np.zeros_like(helmet_rr)
            helmet_nn[:, 2] = 1.
            surf = dict(rr=helmet_rr, nn=helmet_nn,
                        coord_frame=FIFF.FIFFV_COORD_DEVICE)
            helmet_rr = mne.surface.transform_surface_to(
                surf, 'head', meg_info['dev_head_t'])['rr']
            p = np.linspace(0, 2 * np.pi, 40)
            x_sphere, y_sphere = rad * np.sin(p), rad * np.cos(p)
            for ai, ax in enumerate(axs):
                others = np.setdiff1d(np.arange(3), [ai])
                ax.plot(helmet_rr[:, others[0]], helmet_rr[:, others[1]],
                        marker='o', linestyle='none', alpha=0.1,
                        markeredgecolor='none', markerfacecolor='b', zorder=-2)
                ax.plot(x_sphere + r0[others[0]], y_sphere + r0[others[1]],
                        color='y', alpha=0.25, zorder=-1)
                ax.quiver(rr[:, others[0]], rr[:, others[1]],
                          amp * nn[:, others[0]], amp * nn[:, others[1]],
                          angles='xy', units='x', color='k', alpha=0.5)
                ax.set_aspect('equal')
                ax.set_xlabel(' - ' + 'xyz'[others[0]] + ' + ')
                ax.set_ylabel(' - ' + 'xyz'[others[1]] + ' + ')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.setp(list(ax.spines.values()), color='none')
            plt.tight_layout()

    if plot_raw or plot_evoked:
        with printer('Low-pass filtering simulated data'):
            events = mne.find_events(raw_mv, 'STI101', verbose=False)
            b, a = signal.butter(4, 40. / (raw.info['sfreq'] / 2.), 'low',
                                 analog=False)
            raw_mv.filter(None, 40., method='iir', iir_params=dict(b=b, a=a),
                          verbose=False, n_jobs=n_jobs)
        if plot_raw:
            with printer('Plotting raw data'):
                raw_mv.plot(clipping='transparent', events=events,
                            show=False)
        if plot_evoked:
            with printer('Plotting evoked data'):
                picks = mne.pick_types(raw_mv.info, meg=True, eeg=True)
                events[:, 2] = 1
                evoked = mne.Epochs(raw_mv, events, {'Simulated': 1},
                                    0, duration, None, picks).average()
                evoked.plot_topomap(np.unique(tmids), show=False)

    print('\nTotal time: %0.1f sec' % (time.time() - t0))
    sys.stdout.flush()
    if any([plot_dipoles, plot_raw, plot_evoked]):
        plt.show(block=True)


if __name__ == '__main__':
    run()
