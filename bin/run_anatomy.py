# -*- coding: utf-8 -*-
"""Script runs FreeSurfer MRI Reconstruction and returns MNE BEM files for single subject"""

import argparse
import fnmatch
import glob
import os
from os import path as op
from shutil import copy
from subprocess import check_call
from mne import get_config

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2015, ILABS"
__credits__ = ["Kambiz Tavabi"]
__license__ = "BSD (3-clause)"
__version__ = "0"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@gmail.com"
__status__ = "Development"

FS_DIRECTORY = op.join(get_config('SUBJECTS_DIR'))
assert FS_DIRECTORY is not None

# Parse command line call
parser = argparse.ArgumentParser(prog='run_anatomy', description='Run recon-all and create mne BEM files')
parser.add_argument('subject', help='Subject directory')
parser.add_argument('--layers', default=1, help='Define BEM. Defaults to single layer.', type=int)
parser.add_argument('--skip_recon', action='store_true', help='Set to skip FS reconstruction.')
parser.add_argument('--mp', default=8, help='Multi threading argument for recon-all. Default 8.', type=int)
parser.add_argument('--ico', default=4, help='BEM icosahedral parameter for watershed surfaces. Default 4.',
                    type=int)
parser.add_argument('--skip_bem', action='store_true', help='Set to skip BEM computation.')

args = parser.parse_args()

# Set variables
subj = args.subject
bem_type = args.layers
skip_recon = args.skip_recon
skip_bem = args.skip_bem
openmp = args.mp
bem_ico = args.ico
subj_raw_mri_dir = op.join(FS_DIRECTORY, 'DICOM', subj)
subj_dir = op.join(FS_DIRECTORY, subj)
subj_bem_dir = op.join(FS_DIRECTORY, subj, 'bem')
fs_mri_dir = op.join(FS_DIRECTORY, subj, 'mri')
fs_orig_dir = op.join(FS_DIRECTORY, subj, 'mri/orig')
fs_nii_dir = op.join(FS_DIRECTORY, subj, 'mri/nii')
input_rage = op.join(fs_nii_dir, 'MEMPRAGE.nii')

if not skip_recon:
    # TODO(ktavbi@gmail.com): parrec2nii should handle this
    for f in glob.glob(op.join(subj_raw_mri_dir, '*Quiet_Survey*')):
        os.remove(f)
    parfiles = []
    for root, dirnames, filenames in os.walk(subj_raw_mri_dir):
        for filename in fnmatch.filter(filenames, '*.PAR'):
            parfiles.append(os.path.join(root, filename))
    parfiles.sort()
    print('Converting PAREC files for %s ' % subj)
    for ff in parfiles:
        parrec2nii = 'parrec2nii -o %s %s --overwrite' % (subj_raw_mri_dir, ff)
        check_call(parrec2nii.split())

    # Check to see if necessary raw input MR files exist
    input_rage = glob.glob(op.join(subj_raw_mri_dir, '*MEMP_VBM*.nii'))
    if len(input_rage) == 0:
        raise RuntimeError('MPRAGE nifti not found.')
    # Create subject's FS directory
    for folder in [subj_dir, fs_mri_dir, fs_nii_dir, fs_orig_dir]:
        if not op.isdir(folder):
            os.makedirs(folder)

    # copy & link raw nifti files into  subject's FS nii directory and create symlinks
    copy(input_rage[0], fs_nii_dir)
    os.symlink(op.join(fs_nii_dir, op.basename(input_rage[0])),
               op.join(fs_nii_dir, 'MEMPRAGE.nii'))
    # FS rms_concat & recon-all commands
    print('Beginning Freesurfer reconstruction for %s ' % subj)
    rms_concat = 'mri_concat --rms --i %s --o %s/001.mgz' % (input_rage[0], fs_orig_dir)
    recon = 'recon-all -openmp %.0f -subject %s -all' % (openmp, subj)
    # RMS average the multiple echos from the MEMPRAGE
    check_call(rms_concat.split())
    # Do the full reconstruction (about 12 hours)
    check_call(recon.split())

# Start BEM routines
if not skip_bem:
    mne_setup_mri = 'mne_setup_mri mri T1 --subject %s --overwrite' % subj
    check_call(mne_setup_mri.split())
    if bem_type == 3:
        # MEEG BEM from LABSN script
        print('Creating 3-layer BEM for %s ' %subj)
        input_flash5 = glob.glob(op.join(subj_raw_mri_dir, '*FLASH5*.nii'))
        input_flash30 = glob.glob(op.join(subj_raw_mri_dir, '*FLASH30*.nii'))
        for fn, f in zip(['Flash30', 'Flash5'], [input_flash30, input_flash5]):
            if len(f) == 0:
                raise RuntimeError('%s file not found.' % fn)
        copy(input_flash30[0], fs_nii_dir)
        copy(input_flash5[0], fs_nii_dir)
        os.symlink(op.join(fs_nii_dir, op.basename(input_flash30[0])),
                   op.join(fs_nii_dir, 'flash30.nii'))
        os.symlink(op.join(fs_nii_dir, op.basename(input_flash5[0])),
                   op.join(fs_nii_dir, 'flash5.nii'))
        param_maps_dir = op.join(fs_mri_dir, 'flash/parameter_maps')
        if not op.isdir(param_maps_dir):
            os.makedirs(param_maps_dir)
        os.chdir(param_maps_dir)
        for fn in ('flash5', 'flash30'):
            mri_convert = 'mri_convert %s/%s.nii %s.mgz' % (fs_nii_dir, fn, fn)
            check_call(mri_convert.split())
            fsl_rigid_register = 'fsl_rigid_register -r %s/rawavg.mgz -i %s.mgz -o %s_reg.mgz' % (fs_mri_dir, fn, fn)
            check_call(fsl_rigid_register.split())
            if not op.isdir(op.join(fs_mri_dir, '%s' % fn)):
                os.mkdir(op.join(fs_mri_dir, '%s' % fn))
            mri_convert = 'mri_convert -ot cor %s/%s_reg.mgz %s/%s' % (param_maps_dir, fn, fs_mri_dir, fn)
            check_call(mri_convert.split())
        os.chdir(fs_mri_dir)
        mri_convert = 'mri_convert -ot cor brainmask.mgz brain'
        check_call(mri_convert.split())
        mri_convert = 'mri_convert -ot cor T1.mgz T1'
        mri_make_bem_surfaces = 'mri_make_bem_surfaces %s' % subj
        check_call(mri_make_bem_surfaces.split())
        watershed_bem = 'mne_watershed_bem --subject %s --overwrite' % subj
        check_call(watershed_bem.split())
        os.chdir(subj_bem_dir)
        for srf in ('inner', 'outer'):
            mne_convert_surf = 'mne_convert_surface --tri %s_skull.tri --swap --surfout %s_skull.surf' % (srf, srf)
            check_call(mne_convert_surf.split())
        copy(op.join(subj_dir, 'bem/watershed/%s_outer_skin_surface' % subj),
             op.join(subj_dir, 'bem/outer_skin.surf'))
    else:
        print('Creating single layer BEM for %s ' %subj)
        os.chdir(fs_mri_dir)
        watershed_bem = 'mne_watershed_bem --subject %s --overwrite' % subj
        check_call(watershed_bem.split())
        for srf in ('inner_skull', 'outer_skull', 'outer_skin'):
            os.symlink(op.join(subj_dir, 'bem/watershed/%s_%s_surface' % (subj, srf)),
                       op.join(subj_dir, 'bem/%s.surf' % srf))

    mne_setup_fwd = 'mne_setup_forward_model --surf --ico %.0f --subject %s' % (bem_ico, subj)
    check_call(mne_setup_fwd.split())

# Create dense head surface
if not op.isfile(op.join(subj_dir, 'bem/%s-head-dense.fif' % subj)):
    print('Making dense head...')
    mkheadsurf = 'mkheadsurf -subjid %s' % subj
    check_call(mkheadsurf.split())
    os.chdir(subj_bem_dir)
    mne_surf2bem_ = 'mne_surf2bem --surf %s/surf/lh.seghead --id 4 --check --fif %s-head-dense.fif' % (subj_dir, subj)
    check_call(mne_surf2bem_.split())
    if op.isfile(op.join(subj_bem_dir, '%s-head.fif' % subj)):
        os.rename(op.join(subj_bem_dir, '%s-head.fif' % subj), op.join(subj_bem_dir, '%s-head-sparse.fif' % subj))
    os.symlink(op.join(subj_bem_dir, '%s-head-dense.fif' % subj),
               op.join(subj_bem_dir, '%s-head.fif' % subj))
print('Done')
