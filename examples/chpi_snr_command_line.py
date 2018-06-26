# -*- coding: utf-8 -*-
"""
==================================
Demo of cHPI SNR from command line
==================================

Shows how to plot cHPI SNR from the command line. Typically
the command would be called from the command line instead of Python.
"""

import os

print(__doc__)

# make sure we're in the correct path
os.chdir(os.path.abspath(os.path.dirname(__file__)))

cmd = """
../bin/plot_chpi_snr.py funloc/subj_01/raw_fif/subj_01_funloc_raw.fif \
                 --winlen 2 \
                 --nharm 3
"""
os.system(cmd)
