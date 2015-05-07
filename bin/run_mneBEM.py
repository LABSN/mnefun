# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

# TODO(ktavabi@gmail.com): Document
'''Runs FreeSurfer recon-all on RMS combined multi echo MPRAGE volume.

 example usage: python run_mneBEM --subject subject --layers 1
'''
from __future__ import print_function

import sys
import mne
from mne.utils import run_subprocess, logger
import os
from os import path as op
import copy

def run():
    from mne.commands.utils import get_optparser
    
    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    
    parser.add_option('-s', '--subject', dest='subject',
                      help='Freesurfer subject id', type='str')
    parser.add_option('-l', '--layers', dest='layers', default=1, type=int,
                      help='Number BEM layers.')
    parser.add_option('-d', '--subjects-dir', dest='subjects_dir',
                      help='FS Subjects directory', default=subjects_dir)
    
    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    _run(subjects_dir, subject, options.layers)


def _run(subjects_dir, subject, layers):
    this_env = copy.copy(os.environ)
    this_env['SUBJECTS_DIR'] = subjects_dir
    this_env['SUBJECT'] = subject

    if 'SUBJECTS_DIR' not in this_env:
        raise RuntimeError('The environment variable SUBJECTS_DIR should '
                           'be set')

    if not op.isdir(subjects_dir):
        raise RuntimeError('subjects directory %s not found, specify using '
                           'the environment variable SUBJECTS_DIR or '
                           'the command line option --subjects-dir')

    if 'FREESURFER_HOME' not in this_env:
        raise RuntimeError('The FreeSurfer environment needs to be set up '
                           'for this script')
    
    logger.info('1. Setting up MRI files and running watershed algorithm...')
    run_subprocess(['mne_setup_mri', '--mri', 'T1', '--subject', subject, '--overwrite'], env=this_env)
    run_subprocess(['mne_watershed_bem', '--subject', subject, '--overwrite'], env=this_env)

    logger.info('2. Setting up %f.0 layer BEM...' % layers)

    if layers == 3:
        flash05 = op.join(subjects_dir, subject, 'mri/nii/FLASH05.nii')
        flash30 = op.join(subjects_dir, subject, 'mri/nii/FLASH30.nii')

        run_subprocess(['mne', 'flash_bem_model', '-s', subject, '-d', subjects_dir,
                        '--flash05', flash05, '--flash30', flash30, '-v'], env=this_env)

is_main = (__name__ == '__main__')
if is_main:
    run()