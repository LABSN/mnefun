#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Average epochs weighted by the signal-to-noise ratio of continuous HPI.
cHPI SNR is an indicator of the distance between the head and sensors, and
therefore is correlated with the signal-to-noise of cortical sources.

"""

import mne
import argparse
from mnefun import chpi_weighted_average


def run():

    """ Default parameters """
    default_nharm = 2  # number of line harmonics to include

    """ Parse command line """
    parser = argparse.ArgumentParser(description='Weighted averaging of '
                                                 'epochs according to cHPI '
                                                 'SNR.')
    help_snr_file = ('Name of raw fiff file. cHPI SNR will be computed from '
                     'this file. Epochs for averaging will be taken from '
                     'the same file, unless --epochs_file is specified.')
    help_epochs_file = ('Raw fiff file to get the epochs from. It must '
                        'have the same epochs/categories as snr_file.')
    help_reject = ('Whether to use the rejection parameters defined in '
                   'the data acquisition when averaging epochs.')
    help_nharm = 'Number of line frequency harmonics to include.'
    help_epoch_start = 'Epoch start relative to trigger (s)'
    help_epoch_end = 'Epoch end relative to trigger (s)'
    help_plot_snr = 'Whether to plot SNR or not'
    help_stim_channel = 'Which stimulus channel to scan for events'
    help_sti_mask = 'Mask to apply to the stim channel'
    parser.add_argument('snr_file', help=help_snr_file)
    parser.add_argument('--epochs_file', type=str, default=None,
                        help=help_epochs_file)
    parser.add_argument('--reject', help=help_reject, action='store_true')
    parser.add_argument('--nharm', type=int, default=default_nharm,
                        choices=[0, 1, 2, 3, 4], help=help_nharm)
    parser.add_argument('--epoch_start', type=float, default=None,
                        help=help_epoch_start)
    parser.add_argument('--epoch_end', type=float, default=None,
                        help=help_epoch_end)
    parser.add_argument('--plot_snr', help=help_plot_snr, action='store_true')
    parser.add_argument('--stim_channel', help=help_stim_channel, type=str,
                        default=None)
    parser.add_argument('--stim_mask', type=int, default=None,
                        help=help_sti_mask)

    args = parser.parse_args()

    chpi_weighted_average(snr_file,



if __name__ == '__main__':
    run()


