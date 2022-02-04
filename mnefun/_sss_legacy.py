"""Legacy code for running SSS remotely."""
import os.path as op
import shutil
import subprocess

import numpy as np
from mne.bem import fit_sphere_to_headshape
from mne.io import read_info
from mne.io.constants import FIFF
from mne.utils import run_subprocess

from ._paths import _prebad, get_raw_fnames, safe_inserter


def push_raw_files(p, subjects, run_indices):
    """Push raw files to SSS workstation"""
    from ._sss import calc_median_hp
    if len(subjects) == 0:
        return
    print('  Pushing raw files to SSS workstation...')
    # do all copies at once to avoid multiple logins
    shutil.copy2(op.join(op.dirname(__file__), 'run_sss.sh'), p.work_dir)
    includes = ['--include', op.sep + 'run_sss.sh']
    if not isinstance(p.trans_to, str):
        raise TypeError(' Illegal head transformation argument to MaxFilter.')
    elif p.trans_to not in ('default', 'median'):
        _check_trans_file(p)
        includes += ['--include', op.sep + p.trans_to]
    for si, subj in enumerate(subjects):
        subj_dir = op.join(p.work_dir, subj)
        raw_dir = op.join(subj_dir, p.raw_dir)

        out_pos = op.join(raw_dir, subj + '_center.txt')
        if not op.isfile(out_pos):
            print('    Determining head center for %s... ' % subj, end='')
            in_fif = op.join(raw_dir,
                             safe_inserter(p.run_names[0], subj) +
                             p.raw_fif_tag)
            if p.dig_with_eeg:
                dig_kinds = (FIFF.FIFFV_POINT_EXTRA, FIFF.FIFFV_POINT_LPA,
                             FIFF.FIFFV_POINT_NASION, FIFF.FIFFV_POINT_RPA,
                             FIFF.FIFFV_POINT_EEG)
            else:
                dig_kinds = (FIFF.FIFFV_POINT_EXTRA,)
            origin_head = fit_sphere_to_headshape(read_info(in_fif),
                                                  dig_kinds=dig_kinds,
                                                  units='mm')[1]
            out_string = ' '.join(['%0.0f' % np.round(number)
                                   for number in origin_head])
            with open(out_pos, 'w') as fid:
                fid.write(out_string)

        med_pos = op.join(raw_dir, subj + '_median_pos.fif')
        if not op.isfile(med_pos):
            calc_median_hp(p, subj, med_pos, run_indices[si])
        root = op.sep + subj
        raw_root = op.join(root, p.raw_dir)
        includes += ['--include', root, '--include', raw_root,
                     '--include', op.join(raw_root, op.basename(out_pos)),
                     '--include', op.join(raw_root, op.basename(med_pos))]
        prebad_file = _prebad(p, subj)
        includes += ['--include',
                     op.join(raw_root, op.basename(prebad_file))]
        fnames = get_raw_fnames(p, subj, 'raw', True, True, run_indices[si])
        assert len(fnames) > 0
        for fname in fnames:
            assert op.isfile(fname), fname
            includes += ['--include', op.join(raw_root, op.basename(fname))]
    assert ' ' not in p.sws_dir
    assert ' ' not in p.sws_ssh
    cmd = (['rsync', '-aLve', 'ssh -p %s' % p.sws_port, '--partial'] +
           includes + ['--exclude', '*'])
    cmd += ['.', '%s:%s' % (p.sws_ssh, op.join(p.sws_dir, ''))]
    run_subprocess(cmd, cwd=p.work_dir,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def fetch_sss_files(p, subjects, run_indices):
    """Pull SSS files (only designed for *nix platforms)"""
    if len(subjects) == 0:
        return
    includes = []
    for subj in subjects:
        includes += ['--include', subj,
                     '--include', op.join(subj, 'sss_fif'),
                     '--include', op.join(subj, 'sss_fif', '*'),
                     '--include', op.join(subj, 'sss_log'),
                     '--include', op.join(subj, 'sss_log', '*')]
    assert ' ' not in p.sws_dir
    assert ' ' not in p.sws_ssh
    cmd = (['rsync', '-ave', 'ssh -p %s' % p.sws_port, '--partial', '-K'] +
           includes + ['--exclude', '*'])
    cmd += ['%s:%s' % (p.sws_ssh, op.join(p.sws_dir, '*')), '.']
    run_subprocess(cmd, cwd=p.work_dir, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)


def _check_trans_file(p):
    """Helper to make sure our trans_to file exists"""
    if not isinstance(p.trans_to, str):
        raise ValueError('trans_to must be a string')
    if p.trans_to not in ('default', 'median'):
        if not op.isfile(op.join(p.work_dir, p.trans_to)):
            raise ValueError('Trans position file "%s" not found'
                             % p.trans_to)
