# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license


"""Wrapper for setting up mne bem surfaces, creating high resolution skin
surface, and mne source space.

 example usage: python run_mne_bem.py --subject subject
"""
from __future__ import print_function

import sys
import mne
from mne.utils import run_subprocess, logger
import os
from os import path as op
import copy
import shutil


def run():
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')

    parser.add_option('-s', '--subject', dest='subject',
                      help='Freesurfer subject id', type='str')
    parser.add_option('-l', '--layers', dest='layers', default=1, type=int,
                      help='Number BEM layers.')
    parser.add_option('--spacing', dest='spacing', default=5, type=int,
                      help='Triangle decimation number for single layer bem')
    parser.add_option('-d', '--subjects-dir', dest='subjects_dir',
                      help='FS Subjects directory', default=subjects_dir)
    parser.add_option('-o', '--overwrite', dest='overwrite',
                      action='store_true',
                      help='Overwrite existing neuromag MRI and MNE BEM files.')
    options, args = parser.parse_args()

    subject = options.subject
    subjects_dir = options.subjects_dir

    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)

    _run(subjects_dir, subject, options.layers, options.spacing,
         options.overwrite)


def _run(subjects_dir, subject, layers, spacing, overwrite):
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

    subj_path = op.join(subjects_dir, subject)
    if not op.exists(subj_path):
        raise RuntimeError('%s does not exits. Please check your subject '
                           'directory path.' % subj_path)

    logger.info('1. Setting up MRI files...')
    if overwrite:
        run_subprocess(['mne_setup_mri', '--mri', 'T1', '--subject', subject,
                        '--overwrite'], env=this_env)
    else:
        run_subprocess(['mne_setup_mri', '--mri', 'T1', '--subject', subject],
                       env=this_env)

    logger.info('2. Setting up %d layer BEM...' % layers)
    if layers == 3:
        flash05 = op.join(subjects_dir, subject, 'nii/FLASH5.nii')
        flash30 = op.join(subjects_dir, subject, 'nii/FLASH30.nii')

        run_subprocess(
            ['mne', 'flash_bem_model', '-s', subject, '-d', subjects_dir,
             '--flash05', flash05, '--flash30', flash30, '-v'], env=this_env)
        for srf in ('inner_skull', 'outer_skull', 'outer_skin'):
            shutil.copy(
                op.join(subjects_dir, subject, 'bem/flash/%s.surf' % srf),
                op.join(subjects_dir, subject, 'bem/%s.surf' % srf))
    else:
        if overwrite:
            run_subprocess(
                ['mne', 'watershed_bem', '--subject', subject,
                 '--overwrite'], env=this_env)
        else:
            run_subprocess(
                ['mne', 'watershed_bem', '--subject', subject], env=this_env)

    # Create dense head surface and symbolic link to head.fif file
    logger.info(
        '3. Creating high resolution skin surface for coregisteration...')
    run_subprocess(
        ['mne', 'make_scalp_surfaces', '--overwrite', '--subject', subject])
    if op.isfile(op.join(subjects_dir, subject, 'bem/%s-head.fif' % subject)):
        os.rename(op.join(subjects_dir, subject, 'bem/%s-head.fif' % subject),
                  op.join(subjects_dir, subject,
                          'bem/%s-head-sparse.fif' % subject))
    os.symlink(
        (op.join(subjects_dir, subject, 'bem/%s-head-dense.fif' % subject)),
        (op.join(subjects_dir, subject, 'bem/%s-head.fif' % subject)))

    # Create source space
    logger.info(
        '3. Creating mne source space...')
    run_subprocess(['mne_setup_source_space', '--subject', subject, '--spacing',
                    '%.0f' % spacing, '--cps'],
                   env=this_env)


is_main = (__name__ == '__main__')
if is_main:
    run()
