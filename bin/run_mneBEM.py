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
import shutil

def run():
    from mne.commands.utils import get_optparser
    
    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    
    parser.add_option('-s', '--subject', dest='subject',
                      help='Freesurfer subject id', type='str')
    parser.add_option('-l', '--layers', dest='layers', default=1, type=int,
                      help='Number BEM layers.')
    parser.add_option('-i', '--ico', dest='ico', default=4, type=int,
                      help='Triangle decimation number for single layer bem')
    parser.add_option('-d', '--subjects-dir', dest='subjects_dir',
                      help='FS Subjects directory', default=subjects_dir)
    
    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    _run(subjects_dir, subject, options.layers, options.ico)


def _run(subjects_dir, subject, layers, ico):
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
    
    logger.info('1. Setting up MRI files and running watershed algorithm...')
    run_subprocess(['mne_setup_mri', '--mri', 'T1', '--subject', subject, '--overwrite'], env=this_env)
    run_subprocess(['mne_watershed_bem', '--subject', subject, '--overwrite'], env=this_env)

    logger.info('2. Setting up %d layer BEM...' % layers)
    if layers == 3:
        flash05 = op.join(subjects_dir, subject, 'nii/FLASH5.nii')
        flash30 = op.join(subjects_dir, subject, 'nii/FLASH30.nii')

        run_subprocess(['mne', 'flash_bem_model', '-s', subject, '-d', subjects_dir,
                        '--flash05', flash05, '--flash30', flash30, '-v'], env=this_env)
        for srf in ('inner_skull', 'outer_skull', 'outer_skin'):
            shutil.copy(op.join(subjects_dir, subject, 'bem/flash/%s.surf' % srf),
                        op.join(subjects_dir, subject, 'bem/%s.surf' % srf))
        run_subprocess(['mne_setup_forward_model', '--subject', subject, '--ico', '%.0f' % ico, '--surf'],
                       env=this_env)
    else:
        os.symlink(op.join(subjects_dir, subject, 'bem/watershed/%s_inner_skull_surface' % subject),
                   op.join(subjects_dir, subject, 'bem/%s.surf' % 'inner_skull'))
        run_subprocess(['mne_setup_forward_model', '--subject', subject, '--ico', '%.0f' % ico, '--surf', '--homog'],
                       env=this_env)

    # Create dense head surface and symbolic link to head.fif file
    logger.info('3. Creating high resolution skin surface for coregisteration...')
    run_subprocess(['mne', 'make_scalp_surfaces', '--overwrite', '--subject', subject])
    if op.isfile(op.join(subjects_dir, subject, 'bem/%s-head.fif' % subject)):
        os.remove(op.join(subjects_dir, subject, 'bem/%s-head.fif' % subject))
    os.symlink((op.join(subjects_dir, subject, 'bem/%s-head-dense.fif' % subject)),
               (op.join(subjects_dir, subject, 'bem/%s-head.fif' % subject)))

is_main = (__name__ == '__main__')
if is_main:
    run()