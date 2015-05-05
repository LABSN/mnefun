# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

"""Runs FreeSurfer recon-all on RMS combined multi echo MPRAGE volume.

 example usage: mne run_anatomy --subj subject --raw-dr ${SUBJECTS_DIR}/PARREC --openmp 2
"""
from __future__ import print_function

import sys
import mne
from mne.utils import run_subprocess, logger
import fnmatch
import glob
import os
from os import path as op
from shutil import copy


def run():
    from mne.commands.utils import get_optparser
    
    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    
    parser.add_option("-s", "--subj", dest="subj",
                      help="Freesurfer subject id", metavar="FILE")
    parser.add_option("-r", "--raw-dir", dest="raw-dir",
                      help="Path to parent directory containing raw mri data", default="PARREC")
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=subjects_dir)
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='Force FreeSurfer reconstruction.')
    parser.add_option('-o', '--openmp', dest='openmp', default=2,
                      help='Force FreeSurfer reconstruction.')
    
    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    raw_dir = options.raw_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    _run(subjects_dir, subject, raw_dir, options.force, options.openmp)


@verbose
def _run(subjects_dir, subject, raw_dir, force, mp, verbose=None):
    this_env = copy.copy(os.environ)
    this_env['SUBJECTS_DIR'] = subjects_dir
    this_env['SUBJECT'] = subject
    parrec_dir = op.join(subjects_dir, subject, raw_dir)
    
    if 'SUBJECTS_DIR' not in this_env:
        raise RuntimeError('The environment variable SUBJECTS_DIR should '
                           'be set')

    if not op.isdir(subjects_dir):
        raise RuntimeError('subjects directory %s not found, specify using '
                           'the environment variable SUBJECTS_DIR or '
                           'the command line option --subjects-dir')
    
    if not op.isdir(parrec_dir):
        raise RuntimeError('subjects raw data directory %s not found, specify using '
                           'the command line option --raw-dir')

    if 'FREESURFER_HOME' not in this_env:
        raise RuntimeError('The FreeSurfer environment needs to be set up '
                           'for this script')
    
    logger.info('1. Processing raw MRI data with parrec2nii...')
    # TODO(ktavabi@gmail.com): parrec2nii should handle this
    for ff in glob.glob(op.join(parrec_dir, '*Quiet_Survey*')):
        os.remove(ff)
    parrec_files = []
    for root, dirnames, filenames in os.walk(parrec_dir):
        for filename in fnmatch.filter(filenames, '*.PAR'):
            parrec_files.append(op.join(root, filename))
    parrec_files.sort()
    for pf in parrec_files:
        run_subprocess(['parrec2nii', '-o', pf, '--overwrite'], env=this_env)
        
    logger.info('2. Checking to see if raw MPRAGE file exists...')
    input_rage = glob.glob(op.join(parrec_dir, '*MEMP_VBM*.nii'))
    if len(input_rage) == 0:
        raise RuntimeError('%s not found. Please check your '
                           'subject raw directory.' % input_rage)
    
    logger.info('3. Starting FreeSurfer reconstruction process...')
    if op.isdir(op.join(subjects_dir, subject)) and not force:
        raise RuntimeError('%s FreeSurfer directory exists. '
                           'Use command line option --force to overwrite '
                           'previous reconstruction results.' % subject )
    os.mkdir(op.join(subjects_dir, subject))
    run_subprocess(['mri_concat --rms --i', input_rage[0],
                    '--o', op.join(subjects_dir, subject, 'mri/orig/001.mgz')],
                   env=this_env)
    run_subprocess(['recon-all -openmp %.0f -subject', subject, '-all' % mp], env=this_env)
