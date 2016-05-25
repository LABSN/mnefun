# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

"""Wrapper for FreeSurfer reconstruction routines on multi echo MPRAGE or
FLAHS volumes.
example usage: python run_fs_recon --subject subject --raw-dir path-to-raw
"""
from __future__ import print_function
import sys
import mne
from mne.utils import run_subprocess, logger
import fnmatch
import glob
import os
from os import path as op
import copy
import shutil
import nibabel


def run():
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')

    parser.add_option('-s', '--subject', dest='subject',
                      help='Freesurfer subject id', type='str')
    parser.add_option('-r', '--raw-dir', dest='raw_dir',
                      help='Path to parent directory containing raw mri data',
                      default='PARREC', metavar='FILE')
    parser.add_option('-d', '--subjects-dir', dest='subjects_dir',
                      help='FS Subjects directory', default=subjects_dir)
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='Force FreeSurfer reconstruction.')
    parser.add_option('-o', '--openmp', dest='openmp', default=2, type=int,
                      help='Number of CPUs to use for reconstruction routines.')
    parser.add_option('-v', '--volume', dest='volume', default='MPRAGE',
                      type=str,
                      help='Input raw volume file for nii conversion.\n'
                           'Default is MPRAGE, it can also be MEMP.')

    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    raw_dir = options.raw_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    _run(subjects_dir, subject, raw_dir, options.force, options.openmp,
         options.volume)


def _run(subjects_dir, subject, raw_dir, force, mp, volume):
    this_env = copy.copy(os.environ)
    this_env['SUBJECTS_DIR'] = subjects_dir
    this_env['SUBJECT'] = subject
    parrec_dir = op.join(subjects_dir, raw_dir, subject)

    if 'SUBJECTS_DIR' not in this_env:
        raise RuntimeError('The environment variable SUBJECTS_DIR should '
                           'be set')

    if not op.isdir(subjects_dir):
        raise RuntimeError('subjects directory %s not found, specify using '
                           'the environment variable SUBJECTS_DIR or '
                           'the command line option --subjects-dir')

    if not op.isdir(parrec_dir):
        raise RuntimeError('%s directory not found, specify using '
                           'the command line option --raw-dir' % parrec_dir)

    if 'FREESURFER_HOME' not in this_env:
        raise RuntimeError('The FreeSurfer environment needs to be set up '
                           'for this script')

    if op.isdir(op.join(subjects_dir, subject)) and not force:
        raise RuntimeError('%s FreeSurfer directory exists. '
                           'Use command line option --force to overwrite '
                           'previous reconstruction results.' % subject)
    if force:
        shutil.rmtree(op.join(subjects_dir, subject))

    os.mkdir(op.join(subjects_dir, subject))
    os.makedirs(op.join(subjects_dir, subject, 'mri/orig/'))
    os.mkdir(op.join(subjects_dir, subject, 'mri/nii'))
    fs_nii_dir = op.join(subjects_dir, subject, 'mri/nii')

    logger.info('1. Processing raw MRI data...')
    for root, _, filenames in os.walk(parrec_dir):
        for filename in fnmatch.filter(filenames, '*Quiet_Survey*'):
            os.remove(op.join(root, filename))
    parfiles = []
    for root, dirnames, filenames in os.walk(parrec_dir):
        for filename in fnmatch.filter(filenames, '*.PAR'):
            parfiles.append(op.join(root, filename))
    parfiles.sort()
    for pf in parfiles:
        if (volume in pf) or ('FLASH' in pf):
            print('Converting {0}'.format(pf))
            pimg = nibabel.load(pf)
            pr_hdr = pimg.header
            raw_data = pimg.dataobj.get_unscaled()
            affine = pr_hdr.get_affine(origin='fov')
            nimg = nibabel.Nifti1Image(raw_data, affine, pr_hdr)
            nimg.to_filename(op.join(parrec_dir, op.basename(pf)[:-4]))
            shutil.copy(nimg.get_filename(), fs_nii_dir)

    for ff in glob.glob(op.join(fs_nii_dir, '*.nii')):
        if volume in op.basename(ff):
            os.symlink(ff, op.join(fs_nii_dir, 'MPRAGE'))
        elif 'FLASH5' in op.basename(ff):
            os.symlink(ff, op.join(fs_nii_dir, 'FLASH5'))
        elif 'FLASH30' in op.basename(ff):
            os.symlink(ff, op.join(fs_nii_dir, 'FLASH30'))

    logger.info('2. Starting FreeSurfer reconstruction process...')
    mri = op.join(fs_nii_dir, 'MPRAGE.nii')
    run_subprocess(['mri_concat', '--rms', '--i', mri,
                    '--o', op.join(subjects_dir, subject, 'mri/orig/001.mgz')],
                   env=this_env)
    run_subprocess(
        ['recon-all', '-openmp', '%.0f' % mp, '-subject', subject, '-all'],
        env=this_env)
    for morph_to in ['fsaverage', subject]:
        run_subprocess(
            ['mne_make_morph_maps', '--to', morph_to, '--from', subject],
            env=this_env)


is_main = (__name__ == '__main__')
if is_main:
    run()
