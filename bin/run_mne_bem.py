# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license


"""Wrapper script for extracting mne bem surface(s) from subjects FS reconstructed
   MRI data, writing out the MNE BEM solution file for the forward calculations,
   and creating high resolution skin surface for MR-MEEG data coregistration.

   example usage: python run_mne_bem.py --subject subject

   See Also
   --------
   mne.bem.make_flash_bem for options related to using FLASH MRIs for BEM solution.
   mne.make_bem_model
   mne.make_bem_solution

"""
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
    parser.add_option('-o', '--overwrite', dest='overwrite',
                      action='store_true',
                      help='Overwrite existing neuromag MRI and MNE BEM files.')
    parser.add_option('--ico', dest='ico', default=4, type='int',
                      help='The surface ico downsampling to use, e.g. 5=20484, '
                           '4=5120, 3=1280. If None, no subsampling is applied.'
                           'Default is 4.')
    options, args = parser.parse_args()

    subject = options.subject
    subjects_dir = options.subjects_dir

    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)

    _run(subjects_dir, subject, options.layers, options.overwrite, options.ico)


def _run(subjects_dir, subject, layers, overwrite, ico):
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
    os.chdir(subjects_dir)
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

    # Create BEM solution for forward calculations
    logger.info('2. Setting up %d layer BEM...' % layers)
    srf_ds = {5: '20484', 4: '5120', 3: '1280', None: 'None'}
    key = srf_ds.get(ico)
    if layers == 3:
        maps_dir = op.join(subjects_dir, subject, 'mri',
                           'flash', 'parameter_maps')
        os.chdir(maps_dir)
        run_subprocess(
            ['mne', 'flash_bem', '--subject', subject, '--noconvert'],
            env=this_env)
        bem_surf = mne.make_bem_model(subject=subject, ico=ico,
                                       subjects_dir=subjects_dir)
        bem_fname = subject + '-%s-%s-%s-' % (key, key, key) + 'bem-sol.fif'
    else:
        if overwrite:
            run_subprocess(
                ['mne', 'watershed_bem', '--subject', subject,
                 '--overwrite'], env=this_env)
        else:
            run_subprocess(
                ['mne', 'watershed_bem', '--subject', subject], env=this_env)
        # single layer homologous BEM
        bem_surf = mne.make_bem_model(subject=subject, conductivity=[0.3],
                                      ico=ico, subjects_dir=subjects_dir)
        bem_fname = subject + '-%s-' % key + 'bem-sol.fif'
    bem = mne.make_bem_solution(surfs=bem_surf)
    mne.write_bem_solution(fname=op.join(subjects_dir, subject, 'bem',
                                         bem_fname), bem=bem)

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
is_main = (__name__ == '__main__')
if is_main:
    run()
