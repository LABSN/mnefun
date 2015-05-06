# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

"""Runs FreeSurfer recon-all on RMS combined multi echo MPRAGE volume.

 example usage: python run_recon-all --subject subject --raw-dir ${SUBJECTS_DIR}/PARREC --openmp 2
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

def run():
    from mne.commands.utils import get_optparser
    
    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    
    parser.add_option("-s", "--subject", dest="subject",
                      help="Freesurfer subject id", type='str')
    parser.add_option("-r", "--raw-dir", dest="raw_dir",
                      help="Path to parent directory containing raw mri data", default="PARREC", metavar="FILE")
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="FS Subjects directory", default=subjects_dir)
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='Force FreeSurfer reconstruction.')
    parser.add_option('-o', '--openmp', dest='openmp', default=2,
                      help='Number of CPUs to use for reconstruction routines.')
    
    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    raw_dir = options.raw_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    _run(subjects_dir, subject, raw_dir, options.force, options.openmp)


def _run(subjects_dir, subject, raw_dir, force, mp):
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
    
    logger.info('1. Processing raw MRI data with parrec2nii...')
    # TODO(ktavabi@gmail.com): parrec2nii should handle this
    for root, _, filenames in os.walk(parrec_dir):
        for filename in fnmatch.filter(filenames, '*Quiet_Survey*'):
            os.remove(op.join(root, filename))
    parrec_files = []
    for root, dirnames, filenames in os.walk(parrec_dir):
        for filename in fnmatch.filter(filenames, '*.PAR'):
            parrec_files.append(op.join(root, filename))
    parrec_files.sort()
    """for pf in parrec_files:
        run_subprocess(['parrec2nii', '-o', parrec_dir, pf, '--overwrite'], env=this_env)"""

    logger.info('2. Checking to see if raw MPRAGE file exists...')
    input_rage = glob.glob(op.join(parrec_dir, '*MEMP_VBM*.nii'))
    if len(input_rage) == 0:
        raise RuntimeError('%s not found. Please check your '
                           'subject raw directory.' % input_rage[0])
    
    logger.info('3. Starting FreeSurfer reconstruction process...')
    if op.isdir(op.join(subjects_dir, subject)) and not force:
        raise RuntimeError('%s FreeSurfer directory exists. '
                           'Use command line option --force to overwrite '
                           'previous reconstruction results.' % subject)
    if force:
        shutil.rmtree(op.join(subjects_dir, subject))
    os.makedirs(op.join(subjects_dir, subject, 'mri/orig/'))
    run_subprocess(['mri_concat', '--rms', '--i', input_rage[0],
                    '--o', op.join(subjects_dir, subject, 'mri/orig/001.mgz')],
                   env=this_env)
    run_subprocess(['recon-all', '-openmp', mp, '-subject', subject, '-all'], env=this_env)

is_main = (__name__ == '__main__')
if is_main:
    run()
