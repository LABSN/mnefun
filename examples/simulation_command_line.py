# -*- coding: utf-8 -*-
"""
==================================
Demo of the command-line interface
==================================

Show how to use the command-line interface to simulation. Typically
the command would be called from the command line instead of Python.
"""

import os

print(__doc__)

# make sure we're in the correct path
os.chdir(os.path.abspath(os.path.dirname(__file__)))

cmd = """
simulate_movement.py --raw funloc/subj_01/raw_fif/subj_01_funloc_raw.fif \
                     --pos subj_01_funloc_hp_trunc.txt \
                     --dipoles demo_activations.txt \
                     --duration 1 -p
"""
os.system(cmd)
