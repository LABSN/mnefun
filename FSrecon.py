__author__ = 'kambiz'
from subprocess import check_call as chk
import argparse
import glob
import os
from shutil import copy as cp
from os import path as op

parser = argparse.ArgumentParser(prog='FSrecon', description='Run FS recon and create BEM')
parser.add_argument('subject', help='Subject directory')
parser.add_argument('--layers', default=1, help='Define BEM. Defaults to single layer.',
                    type=int)

args = parser.parse_args()


FS_dir = os.getenv('SUBJECTS_DIR')
assert FS_dir is not None

recon_dir = op.join(FS_dir, 'Recon', args.subject)

INRAGE = glob.glob(op.join(recon_dir, '*MEMP_VBM*.nii'))
if len(INRAGE) == 0:
    raise RuntimeError('MPRAGE nifti not found.')
if args.layers == 3:
    IN5 = glob.glob(op.join(recon_dir, '*FLASH5*.nii'))
    IN30 = glob.glob(op.join(recon_dir, '*FLASH30*.nii'))
    for fn, f in zip(['Flash30', 'Flash5'], [IN30, IN5]):
        if len(f) == 0:
            raise RuntimeError('%s nifiti not found.' % fn)

subj_dir = op.join(FS_dir, args.subject)
if not op.isdir(subj_dir):
    os.mkdir(subj_dir)
if not op.isdir(op.join(FS_dir, args.subject, 'mri')):
    os.mkdir(op.join(FS_dir, args.subject, 'mri'))
if not op.isdir(op.join(FS_dir, args.subject, 'mri', 'orig')):
    os.mkdir(op.join(FS_dir, args.subject, 'mri', 'orig'))
nii_dir = op.join(FS_dir, args.subject, 'nii')
if not op.isdir(nii_dir):
    os.mkdir(op.join(FS_dir, args.subject, 'nii'))
mri_dir = op.join(subj_dir, 'mri')

cp(INRAGE[0], nii_dir)
cp(IN30[0], nii_dir)
cp(IN5[0], nii_dir)

os.symlink(op.join(nii_dir, op.basename(INRAGE[0])), op.join(nii_dir, 'MEMPRAGE.nii'))
os.symlink(op.join(nii_dir, op.basename(IN30[0])), op.join(nii_dir, 'flash30.nii'))
os.symlink(op.join(nii_dir, op.basename(IN5[0])), op.join(nii_dir, 'flash5.nii'))

INRAGE = glob.glob(op.join(nii_dir, 'MEMPRAGE.nii'))
orig_mr_dir = op.join(FS_dir, args.subject, 'mri', 'orig')

rms_concat = 'mri_concat --rms --i %s --o %s/001.mgz' % (INRAGE[0], orig_mr_dir)
recon = 'recon-all -openmp 20 -subject %s -all' % args.subject
# RMS average the multiple echos from the MEMPRAGE
chk(rms_concat.split())
# Do the full reconstruction (about 12 hours)
chk(recon.split())

# Start BEM routines
mne_setup_mri_ = 'mne_setup_mri --mri T1 --SUBJECT %s --overwrite' % args.subject
chk(mne_setup_mri_.split())


if args.layers == 3:
    # MEEG BEM from LABSN script
    print('Creating 3-layer BEM')
    os.chdir(op.join(FS_dir, args.subject))
    source_dir = op.join(subj_dir, 'nii')
    dest_dir = op.join(subj_dir, 'flash', 'parameter_maps')
    os.makedirs(dest_dir)

    os.chdir(dest_dir)
    for fn in 'flash5, flash30':
        mri_convert_ = 'mri_convert %s/%s.nii ./%s.mgz' % (source_dir, fn, fn)
        chk(mri_convert_.split())
        fsl_rigid_register_ = 'fsl_rigid_register -r ../../rawavg.mgz -i ./%s.mgz -o %s_reg.mgz' % (fn, fn)
        chk(fsl_rigid_register_.split())
        os.makedirs(args.subject, 'mri', fn)
        mri_convert_ = 'mri_convert -ot cor ./%s_reg.mz ../../%s' % (fn, fn)
        chk(mri_convert_.split())
    os.chdir(mri_dir)
    mri_convert_ = 'mri_convert -ot cor brainmask.mgz brain'
    chk(mri_convert_.split())
    mri_convert_ = 'mri_convert -ot cor T1.mgz T1'
    mri_make_bem_surfaces_ = 'mri_make_bem_surfaces %s' % args.subject
    chk(mri_make_bem_surfaces_.split())
    watershed_bem_ = 'mne_watershed_bem --subject %s --overwrite' % args.subject
    chk(watershed_bem_.split())
    os.chdir(subj_dir, 'bem')
    for srf in ('inner', 'outer')
        mne_convert_surf_ = 'mne_convert_surface --tri %s_skull.tri --swap --surfout %s_skull.surf' % (srf, srf)
        chk(mne_convert_surf_.split())
    cp(op.join(subj_dir, 'bem', 'watershed', '%s_outer_skin_surface' % args.subject),
       op.join(subj_dir, 'bem', 'outer_skin.surf'))
else:
    # MEEG BEM from LABSN script
    print('Creating single layer BEM')
    os.chdir(mri_dir)
    watershed_bem_ = 'mne_watershed_bem --subject %s --overwrite' % args.subject
    chk(watershed_bem_.split())

    os.symlink(op.join(nii_dir, op.basename(INRAGE[0])), op.join(nii_dir, 'MEMPRAGE.nii'))

    for srf in ('inner_skull', 'outer_skull', 'outer_skin'):
        os.symlink(op.join(subj_dir, 'bem', 'watershed', '%s_%s_surface' % (args.subject, srf)),
                   op.join(subj_dir, 'bem', '%s.surf' % srf))

mne_setup_fwd_ = 'mne_setup_forward_model --surf --ico 4 --subject %s' % args.subject
chk(mne_setup_fwd_.split())

# Create dense head surface
if not op.isfile(op.join(subj_dir, 'bem', '%s-head-dense.fif' % args.subject)):
    print('Making dense head...')
    mkheadsurf_ = 'mkheadsurf -subjid %s' % args.subject
    chk(mkheadsurf_.split())
    os.chdir(op.join(subj_dir, 'bem'))
    mne_surf2bem_ = 'mne_surf2bem --surf ../surf/lh.seghead --id 4 --check --fif %s-head-dense.fif' % args.subject
    chk(mne_surf2bem_.split())
else:
    os.remove(op.join(subj_dir, 'bem', '%s-head.fif' % args.subject))

os.symlink(op.join(subj_dir, 'bem', '%s-head-dense.fif' % args.subject),
           op.join(subj_dir, 'bem', '%s-head.fif' % args.subject))
print('Done')
