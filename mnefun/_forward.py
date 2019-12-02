"""Forward computation."""

import os
import os.path as op
import warnings

import numpy as np
from mne import (read_bem_solution, dig_mri_distances,
                 make_forward_solution, read_source_spaces, make_sphere_model,
                 write_source_spaces, setup_source_space, read_trans,
                 setup_volume_source_space, write_forward_solution)
from mne.io import read_info
from mne.utils import get_subjects_dir

from ._paths import get_raw_fnames, safe_inserter
from ._utils import _handle_dict


def gen_forwards(p, subjects, structurals, run_indices):
    """Generate forward solutions

    Can only complete successfully once coregistration is performed
    (usually in mne_analyze).

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    structurals : list (of str or None)
        The structural data names for each subject (e.g., ['AKCLEE_101', ...]).
        If None, a spherical BEM and volume grid space will be used.
    run_indices : array-like | None
        Run indices to include.
    """
    for si, subj in enumerate(subjects):
        struc = structurals[si]
        fwd_dir = op.join(p.work_dir, subj, p.forward_dir)
        if not op.isdir(fwd_dir):
            os.mkdir(fwd_dir)
        raw_fname = get_raw_fnames(p, subj, 'sss', False, False,
                                   run_indices[si])[0]
        info = read_info(raw_fname)
        bem, src, trans, bem_type = _get_bem_src_trans(p, info, subj, struc)
        if not getattr(p, 'translate_positions', True):
            raise RuntimeError('Not translating positions is no longer '
                               'supported')
        print('  Creating forward solution(s) using a %s for %s...'
              % (bem_type, subj), end='')
        # XXX Don't actually need to generate a different fwd for each inv
        # anymore, since all runs are included, but changing the filename
        # would break a lot of existing pipelines :(
        try:
            subjects_dir = get_subjects_dir(p.subjects_dir, raise_error=True)
            subject = src[0]['subject_his_id']
            dist = dig_mri_distances(info, trans, subject,
                                     subjects_dir=subjects_dir)
        except Exception as exp:
            # old MNE or bad args
            print(' (dig<->MRI unknown: %s)' % (str(exp)[:20] + '...',))
        else:
            dist = np.median(dist)
            print(' (dig<->MRI %0.1f mm)' % (1000 * dist,))
            if dist > 5:
                warnings.warn(
                    '%s dig<->MRI distance %0.1f mm could indicate a problem '
                    'with coregistration, check coreg'
                    % (subject, 1000 * dist))
        for ii, (inv_name, inv_run) in enumerate(zip(p.inv_names,
                                                     p.inv_runs)):
            fwd_name = op.join(fwd_dir, safe_inserter(inv_name, subj) +
                               p.inv_tag + '-fwd.fif')
            fwd = make_forward_solution(
                info, trans, src, bem, n_jobs=p.n_jobs, mindist=p.fwd_mindist)
            write_forward_solution(fwd_name, fwd, overwrite=True)


def _get_bem_src_trans(p, info, subj, struc):
    subjects_dir = get_subjects_dir(p.subjects_dir, raise_error=True)
    assert isinstance(subjects_dir, str)
    if struc is None:  # spherical case
        bem, src, trans = _spherical_conductor(info, subj, p.src_pos)
        bem_type = 'spherical-model'
    else:
        from mne.transforms import _ensure_trans
        trans = op.join(p.work_dir, subj, p.trans_dir, subj + '-trans.fif')
        if not op.isfile(trans):
            old = trans
            trans = op.join(p.work_dir, subj, p.trans_dir,
                            subj + '-trans_head2mri.txt')
            if not op.isfile(trans):
                raise IOError('Unable to find head<->MRI trans files in:\n'
                              '%s\n%s' % (old, trans))
        trans = read_trans(trans)
        trans = _ensure_trans(trans, 'mri', 'head')
        this_src = _handle_dict(p.src, subj)
        assert isinstance(this_src, str)
        if this_src.startswith('oct'):
            kind = 'oct'
        elif this_src.startswith('vol'):
            kind = 'vol'
        else:
            raise RuntimeError('Unknown source space type %s, must be '
                               'oct or vol' % (this_src,))
        num = int(this_src.split(kind)[-1].split('-')[-1])
        bem = op.join(subjects_dir, struc, 'bem', '%s-%s-bem-sol.fif'
                      % (struc, p.bem_type))
        for mid in ('', '-'):
            src_space_file = op.join(subjects_dir, struc, 'bem',
                                     '%s-%s%s%s-src.fif'
                                     % (struc, kind, mid, num))
            if op.isfile(src_space_file):
                break
        else:  # if neither exists, use last filename
            print('    Creating %s%s source space for %s...'
                  % (kind, num, subj))
            if kind == 'oct':
                src = setup_source_space(
                    struc, spacing='%s%s' % (kind, num),
                    subjects_dir=p.subjects_dir, n_jobs=p.n_jobs)
            else:
                assert kind == 'vol'
                src = setup_volume_source_space(
                    struc, pos=num, bem=bem, subjects_dir=p.subjects_dir)
            write_source_spaces(src_space_file, src)
        src = read_source_spaces(src_space_file)
        bem = read_bem_solution(bem, verbose=False)
        bem_type = ('%s-layer BEM' % len(bem['surfs']))
    return bem, src, trans, bem_type


def _spherical_conductor(info, subject, pos):
    """Helper to make spherical conductor model."""
    bem = make_sphere_model(info=info, r0='auto',
                            head_radius='auto', verbose=False)
    src = setup_volume_source_space(sphere=bem, pos=pos, mindist=1.)
    return bem, src, None
