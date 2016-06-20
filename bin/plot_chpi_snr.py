#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot signal-to-noise of continuous HPI coils as a function of time.
Works by fitting a general linear model (HPI freqs, line freqs, DC, slope) to
the data, and comparing estimated HPI powers with the residual (=variance
unexplained by the model).
Window length for SNR estimates can be specified on the command line.
Longer windows will by nature include more low frequencies and thus have
larger residual variance (lower SNR).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import mne
import argparse
from mnefun import plot_chpi_snr_raw


def run():

    # default parameters
    default_winlen = 1  # window length, seconds
    default_nharm = 2  # number of line harmonics to include

    parser = argparse.ArgumentParser()
    parser.add_argument('fiff_file', help='Name of raw fiff file')
    parser.add_argument('--winlen', type=float, default=default_winlen,
                        help='Buffer length for SNR estimates (s)')
    parser.add_argument('--nharm', type=int, default=default_nharm,
                        choices=[0, 1, 2, 3, 4], help='Number of line'
                        'frequency harmonics to include')
    args = parser.parse_args()

    raw = mne.io.Raw(args.fiff_file, allow_maxshield='yes')
    plot_chpi_snr_raw(raw, args.winlen, args.nharm)


if __name__ == '__main__':
    run()
