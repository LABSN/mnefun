#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Average epochs weighted by the signal-to-noise ratio of continuous HPI.
cHPI SNR is an indicator of the distance between the head and sensors, and
therefore is correlated with the signal-to-noise of cortical sources.
"""

from __future__ import print_function

import os.path as op
import matplotlib.pyplot as plt
import mne
import argparse
from mnefun import chpi_weighted_average
from numpy import log10
import sys


def run():

    """ Default parameters """
    default_nharm = 5  # number of line harmonics to include

    """ Parse command line """
    parser = argparse.ArgumentParser(description='Weighted averaging of '
                                                 'epochs according to cHPI '
                                                 'SNR.')
    help_snr_file = ('Name of raw fiff file. cHPI SNR will be computed from '
                     'this file. Epochs for averaging will be taken from '
                     'the same file, unless --epochs_file is specified.')
    help_epochs_file = ('Raw fiff file to get the epochs from. It must '
                        'have the same epochs/categories as snr_file.')
    help_reject = ('Whether to use the rejection criteria defined in '
                   'the data acquisition when averaging epochs.')
    help_flat = ('Whether to use the flatness criteria defined in '
                 'the data acquisition when averaging epochs.')
    help_nharm = 'Number of line frequency harmonics to include.'
    help_epoch_start = 'Epoch start relative to trigger (s)'
    help_epoch_end = 'Epoch end relative to trigger (s)'
    help_plot_snr = 'Whether to plot SNR or not'
    help_stim_channel = 'Which stimulus channel to scan for events'
    help_mask = 'Mask to apply to the stim channel'
    parser.add_argument('raw_snr', help=help_snr_file)
    parser.add_argument('--raw_epochs', type=str, default=None,
                        help=help_epochs_file)
    parser.add_argument('--reject', help=help_reject, action='store_true')
    parser.add_argument('--flat', help=help_flat, action='store_true')
    parser.add_argument('--nharm', type=int, default=default_nharm,
                        help=help_nharm)
    parser.add_argument('--epoch_start', type=float, default=None,
                        help=help_epoch_start)
    parser.add_argument('--epoch_end', type=float, default=None,
                        help=help_epoch_end)
    parser.add_argument('--plot_snr', help=help_plot_snr, action='store_true')
    parser.add_argument('--stim_channel', help=help_stim_channel, type=str,
                        default=None)
    parser.add_argument('--mask', type=int, default=None, help=help_mask)

    args = parser.parse_args()

    if args.raw_epochs:
        fnbase = op.basename(op.splitext(args.raw_epochs)[0])
    else:
        fnbase = op.basename(op.splitext(args.raw_snr)[0])
    fname_avg = fnbase + '_chpi_weighted-ave.fif'
    if op.isfile(fname_avg):
        print('Output file %s already exists!' % fname_avg)
        sys.exit()

    mne.set_log_level('ERROR')  # reduce mne output
    verbose = False

    # cHPI file is typically not maxfiltered, so ignore MaxShield warnings
    raw_snr = mne.io.Raw(args.raw_snr, allow_maxshield='yes',
                         verbose=verbose)

    # if using a separate file for the actual data epochs, load it too
    if args.raw_epochs:
        raw_epochs = mne.io.Raw(args.raw_epochs, allow_maxshield=True,
                                verbose=verbose)
    else:
        raw_epochs = None

    cargs = args.__dict__.copy()
    cargs.pop('plot_snr')  # do not pass these arguments
    cargs.pop('raw_epochs')
    cargs.pop('raw_snr')
    evokeds, snrs = chpi_weighted_average(raw_snr, raw_epochs=raw_epochs,
                                          **cargs)
    nevo = len(evokeds)
    if args.plot_snr:
        plt.figure()
        for k, w_snr in enumerate(snrs):
            plt.subplot(nevo, 1, k+1)
            plt.plot(10*log10(w_snr))
            plt.title('Per-epoch cHPI SNR for category: ' + evokeds[k].comment)
            plt.xlabel('Index of epoch (good epochs only)')
            plt.ylabel('cHPI SNR (dB)')
        plt.tight_layout()
        plt.show()

    # write all resulting evoked objects to a fiff file """
    if evokeds:
        print('Writing %s' % fname_avg)
        mne.write_evokeds(fname_avg, evokeds)


if __name__ == '__main__':
    run()
