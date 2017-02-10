# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

"""Wrapper for FreeSurfer reconstruction routines on multi echo MPRAGE or FLASH volumes.

    example usage: python run_fs_recon -s subject

    Notes
    -----
    This script assumes that the raw structural MRI are saved in PARREC format.
    Before running this script do the following:

        1. Make sure your freesurfer subjects directory is setup such that
           a sub-directory e.g., DICOM contains subject raw MRI data.
        2. Raw parrec file names must contain following minimal substrings for
           script to work with minimal invocation: FLASH5, FLASH30, MPRAGE for
           FLASH MRIs with flip angles 5 & 30, and T1 weighted structural.
"""
from __future__ import print_function
import sys
import mne
from mne.utils import run_subprocess, logger
import fnmatch
import os
from os import path as op
import copy
import shutil
import re
import nibabel


def process_flash(mri_dir, raw_dir, nii_dir, subj_id):
    """helper to convert FLASH MRIs to mgz format for creating inner/outer skull surfaces """
    maps_dir = op.join(mri_dir, 'flash', 'parameter_maps')
    os.makedirs(maps_dir)
    # find flash parrecs convert to nii and copy to Fs nii dir
    flash_files = find_files(raw_dir, '*FLASH*')
    # flip angle 30
    regexp = re.compile(r'.*(FLASH30).*.(PAR)')
    for f in flash_files:
        if regexp.search(f):
            flash30_parrec = f
    parrec_to_nii(flash30_parrec, nii_dir)
    flash30_nii = op.join(nii_dir, op.basename(flash30_parrec[:-4] + '.nii'))
    run_subprocess(['mri_convert', flash30_nii,
                    op.join(mri_dir, maps_dir, 'flash30.mgz')])
    run_subprocess(['fsl_rigid_register',
                    '-r', op.join(mri_dir, 'rawavg.mgz'),
                    '-i', op.join(maps_dir, 'flash30.mgz'), '-o',
                    op.join(maps_dir, 'flash30_reg.mgz')])
    os.mkdir(op.join(mri_dir, 'flash30'))
    run_subprocess(['mri_convert', '-ot', 'cor',
                    op.join(maps_dir, 'flash30_reg.mgz'),
                    op.join(mri_dir, 'flash30')])
    # flip angle 5
    regexp = re.compile(r'.*(FLASH5).*.(PAR)')
    for f in flash_files:
        if regexp.search(f):
            flash5_parrec = f
    parrec_to_nii(flash5_parrec, nii_dir)
    flash5_nii = op.join(nii_dir, op.basename(flash5_parrec[:-4] + '.nii'))
    run_subprocess(['mri_convert', flash5_nii,
                    op.join(mri_dir, maps_dir, 'flash5.mgz')])
    run_subprocess(['fsl_rigid_register',
                    '-r', op.join(mri_dir, 'rawavg.mgz'),
                    '-i', op.join(maps_dir, 'flash5.mgz'), '-o',
                    op.join(maps_dir, 'flash5_reg.mgz')])
    os.mkdir(op.join(mri_dir, 'flash5'))
    run_subprocess(['mri_convert', '-ot', 'cor',
                    op.join(maps_dir, 'flash5_reg.mgz'),
                    op.join(mri_dir, 'flash5')])

    run_subprocess(['mri_convert', '-ot', 'cor',
                    op.join(mri_dir, 'brainmask.mgz'), 'brain'])
    run_subprocess(['mri_convert', '-ot', 'cor',
                    op.join(mri_dir, 'T1.mgz'), 'T1'])


def find_files(dir, pattern):
    """helper to find specific files and return sorted list"""
    matches = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(op.join(root, filename))
    return sorted(matches)


def parrec_to_nii(file_in, out_put_dir):
    """helper to convert PARREC to nifti format"""
    print('Converting {0}'.format(file_in))
    pimg = nibabel.load(file_in)
    pr_hdr = pimg.header
    raw_data = pimg.dataobj.get_unscaled()
    affine = pr_hdr.get_affine(origin='fov')
    nimg = nibabel.Nifti1Image(raw_data, affine, pr_hdr)
    nimg.to_filename(op.basename(file_in)[:-4])
    shutil.copy(nimg.get_filename(), out_put_dir)


def run():
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')

    parser.add_option('-s', '--subject', dest='subject',
                      help='Freesurfer subject id', type='str')
    parser.add_option('-r', '--raw-dir', dest='raw_dir',
                      help='Path to parent directory containing raw mri data',
                      default='DICOM', metavar='FILE')
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
    parser.add_option('--flash', dest='flash', action='store_true',
                      help='If true then find and process flash MR files.')

    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    raw_dir = options.raw_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    _run(subjects_dir, subject, raw_dir, options.force, options.openmp,
         options.volume, options.flash)


def _run(subjects_dir, subject, raw_dir, force, mp, volume, flash):
    this_env = copy.copy(os.environ)
    this_env['SUBJECTS_DIR'] = subjects_dir
    this_env['SUBJECT'] = subject
    dicom_dir = op.join(subjects_dir, raw_dir, subject)

    if 'SUBJECTS_DIR' not in this_env:
        raise RuntimeError('The environment variable SUBJECTS_DIR should '
                           'be set')

    if not op.isdir(subjects_dir):
        raise RuntimeError('subjects directory %s not found, specify using '
                           'the environment variable SUBJECTS_DIR or '
                           'the command line option --subjects-dir')

    if not op.isdir(dicom_dir):
        raise RuntimeError('%s directory not found, specify using '
                           'the command line option --raw-dir' % dicom_dir)

    if 'FREESURFER_HOME' not in this_env:
        raise RuntimeError('The FreeSurfer environment needs to be set up '
                           'for this script')

    if op.isdir(op.join(subjects_dir, subject)) and not force:
        raise RuntimeError('%s FreeSurfer directory exists. '
                           'Use command line option --force to overwrite '
                           'previous reconstruction results.' % subject)
    if force:
        shutil.rmtree(op.join(subjects_dir, subject))

    fs_mri_dir = op.join(subjects_dir, subject, 'mri')
    fs_nii_dir = op.join(fs_mri_dir, 'nii')
    os.makedirs(fs_nii_dir)
    os.mkdir(op.join(fs_mri_dir, 'orig'))

    logger.info('1. Processing T1 MRI...')
    # trash survey scan
    for root, _, filenames in os.walk(dicom_dir):
        for filename in fnmatch.filter(filenames, '*Quiet_Survey*'):
            os.remove(op.join(root, filename))
    t1_parrec = find_files(dicom_dir, '*%s*.PAR' % volume)[0]
    t1_nii = op.join(fs_nii_dir, op.basename(t1_parrec[:-4] + '.nii'))
    parrec_to_nii(t1_parrec, fs_nii_dir)
    logger.info('2. Starting FreeSurfer reconstruction...')
    run_subprocess(['mri_concat', '--rms', '--i', t1_nii,
                    '--o', op.join(subjects_dir, subject,
                                   'mri/orig/001.mgz')], env=this_env)
    run_subprocess(['recon-all', '-openmp', '%.0f' % mp,
                        '-subject', subject, '-all'], env=this_env)

    logger.info('3. Making mne morph map to fsaverage...')
    run_subprocess(
        ['mne_make_morph_maps', '--to', 'fsaverage', '--from', subject],
        env=this_env)

    if flash:
        logger.info('4. Processing FLASH files...')
        process_flash(fs_mri_dir, dicom_dir, fs_nii_dir, subject)

is_main = (__name__ == '__main__')
if is_main:
    run()
