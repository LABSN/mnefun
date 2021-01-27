"""Miscellaneous utilities."""

from functools import reduce
import os
import os.path as op
import shutil
import subprocess
import warnings

import numpy as np
import mne
from mne import (pick_types, pick_info, make_sphere_model, DipoleFixed, Epochs,
                 Dipole, make_forward_dipole, Projection)
from mne.channels import make_standard_montage, make_dig_montage
from mne.fixes import _get_args as get_args  # noqa: F401
from mne.io.constants import FIFF
from mne.utils import verbose


def _fix_raw_eog_cals(raws, kind='EOG'):
    """Fix for annoying issue where EOG cals don't match."""
    # Warning: this will only produce correct EOG scalings with preloaded
    # raw data!
    if kind == 'EOG':
        picks = pick_types(raws[0].info, eeg=False, meg=False, eog=True,
                           exclude=[])
    else:
        assert kind == 'all'
        picks = np.arange(len(raws[0].ch_names))
    if len(picks) > 0:
        first_cals = _cals(raws[0])[picks]
        for ri, r in enumerate(raws[1:]):
            if kind == 'EOG':
                picks_2 = pick_types(r.info, eeg=False, meg=False, eog=True,
                                     exclude=[])
            else:
                picks_2 = np.arange(len(r.ch_names))

            assert np.array_equal(picks, picks_2)
            these_cals = _cals(r)[picks]
            if not np.array_equal(first_cals, these_cals):
                warnings.warn('Adjusting %s cals for %s'
                              % (kind, op.basename(r._filenames[0])))
                _cals(r)[picks] = first_cals


def _cals(raw):
    """Helper to deal with the .cals->._cals attribute change."""
    try:
        return raw._cals
    except AttributeError:
        return raw.cals


def _get_baseline(p):
    """Helper to extract baseline from params."""
    if p.baseline is None:
        return p.baseline
    elif p.baseline == 'individual':
        baseline = (p.bmin, p.bmax)
    else:
        baseline = p.baseline
    # XXX this and some downstream stuff (e.g., tmin=-baseline[0]) won't work
    # for baseline=None, but we can fix that when someone needs it
    # SMB (2020.04.20): added return None to skip baseline application.
    baseline = tuple(baseline)
    if baseline[0] is None:
        baseline = (p.tmin, baseline[1])
    if baseline[1] is None:
        baseline = (baseline[0], p.tmax)
    return baseline


def _handle_dict(entry, subj):
    out = entry
    if isinstance(entry, dict):
        try:
            out = entry[subj]
        except KeyError:
            pass
    return out


def _handle_decim(decim, sfreq):
    decim = np.array(decim)
    assert decim.shape == ()
    if decim.dtype.char in 'il':
        return decim
    else:
        # float
        assert decim.dtype.char == 'd', decim.dtype.char
        got_decim = int(round(sfreq / decim))
        assert np.isclose(sfreq / got_decim, decim), (sfreq, decim, got_decim)
        return got_decim


def _safe_remove(fnames):
    if isinstance(fnames, str):
        fnames = [fnames]
    for fname in fnames:
        if op.isfile(fname):
            os.remove(fname)


def _restrict_reject_flat(reject, flat, raw):
    """Restrict a reject and flat dict based on channel presence"""
    reject = {} if reject is None else reject
    flat = {} if flat is None else flat
    assert isinstance(reject, dict)
    assert isinstance(flat, dict)
    use_reject, use_flat = dict(), dict()
    for in_, out in zip([reject, flat], [use_reject, use_flat]):
        use_keys = [key for key in in_.keys() if key in raw]
        for key in use_keys:
            out[key] = in_[key]
    return use_reject, use_flat


def timestring(t):
    """Reformat time to convenient string

    Parameters
    ----------
    t : float
        Elapsed time in seconds.

    Returns
    time : str
        The time in HH:MM:SS.
    """

    def rediv(ll, b):
        return list(divmod(ll[0], b)) + ll[1:]

    return "%d:%02d:%02d.%03d" % tuple(reduce(rediv, [[t * 1000, ], 1000, 60,
                                                      60]))


def source_script(script_name):
    """Set environmental variables by source-ing a bash script

    Parameters
    ----------
    script_name : str
        Path to the script to execute and get the environment variables from.
    """
    cmd = ['bash', '-c', 'source ' + script_name + ' > /dev/null && env']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in proc.stdout:
        (key, _, value) = line.partition("=")
        os.environ[key] = value.strip()
    proc.communicate()


def make_montage(info, kind, check=False):
    from mne.utils import _clean_names
    import mnefun
    assert kind in ('mgh60', 'mgh70', 'uw_70', 'uw_60')
    picks = pick_types(info, meg=False, eeg=True, exclude=())
    sphere = make_sphere_model('auto', 'auto', info)
    info = pick_info(info, picks)
    to_names = info['ch_names']
    if kind in ('mgh60', 'mgh70'):
        if kind == 'mgh60':
            assert len(to_names) in (59, 60)
        else:
            assert len(to_names) in (70,)
        montage = make_standard_montage(
            kind, head_size=sphere.radius)
        from_names = _clean_names(to_names, remove_whitespace=True)
    else:
        assert len(to_names) == 60
        from_names = getattr(mnefun, 'ch_names_' + kind)
        montage = make_standard_montage(
            'standard_1020', head_size=sphere.radius)
    assert len(from_names) == len(to_names)
    montage_pos = montage._get_ch_pos()
    montage = make_dig_montage(
        {to: montage_pos[fro] for fro, to in zip(from_names, to_names)},
        coord_frame='head')
    eeg_pos = np.array([ch['loc'][:3] for ch in info['chs']])
    montage_pos = montage._get_ch_pos()
    montage_pos = np.array([montage_pos[name] for name in to_names])
    assert len(eeg_pos) == len(montage_pos)
    if check:
        from mayavi import mlab
        mlab.figure(size=(800, 800))
        mlab.points3d(*sphere['r0'], scale_factor=2 * sphere.radius,
                      color=(0., 0., 1.), opacity=0.1, mode='sphere')
        mlab.points3d(*montage_pos.T, scale_factor=0.01,
                      color=(1, 0, 0), mode='sphere', opacity=0.5)
        mlab.points3d(*eeg_pos.T, scale_factor=0.005, color=(1, 1, 1),
                      mode='sphere', opacity=1)
    return montage, sphere


def compute_auc(dip, tmin=-np.inf, tmax=np.inf):
    """Compute the AUC values for a DipoleFixed object."""
    from mne.utils import _time_mask
    if not isinstance(dip, DipoleFixed):
        raise TypeError('dip must be a DipoleFixed, got "%s"' % (type(dip),))
    pick = pick_types(dip.info, meg=False, dipole=True)
    if len(pick) != 1:
        raise RuntimeError('Could not find dipole data')
    time_mask = _time_mask(dip.times, tmin, tmax, dip.info['sfreq'])
    data = dip.data[pick[0], time_mask]
    return np.sum(np.abs(data)) * len(data) * (1. / dip.info['sfreq'])


def _get_epo_kwargs():
    from mne.fixes import _get_args
    epo_kwargs = dict(verbose=False)
    if 'overwrite' in _get_args(Epochs.save):
        epo_kwargs['overwrite'] = True
    return epo_kwargs


@verbose
def make_dipole_projectors(info, pos, ori, bem, trans, verbose=None):
    """Make dipole projectors.

    Parameters
    ----------
    info : instance of Info
        The measurement info.
    pos : ndarray, shape (n_dip, 3)
        The dipole positions.
    ori : ndarray, shape (n_dip, 3)
        The dipole orientations.
    bem : instance of ConductorModel
        The conductor model to use.
    trans : instance of Transform
        The mri-to-head transformation.
    %(verbose)s

    Returns
    -------
    projs : list of Projection
        The projectors.
    """
    pos = np.atleast_2d(pos).astype(float)
    ori = np.atleast_2d(ori).astype(float)
    if pos.shape[1] != 3 or pos.shape != ori.shape:
        raise ValueError('pos and ori must be 2D, the same shape, and have '
                         f'last dimension 3, got {pos.shape} and {ori.shape}')
    dip = Dipole(
        pos=pos, ori=ori, amplitude=np.ones(pos.shape[0]),
        gof=np.ones(pos.shape[0]), times=np.arange(pos.shape[0]))
    info = pick_info(info, pick_types(info, meg=True, eeg=True, exclude=()))
    fwd, _ = make_forward_dipole(dip, bem, info, trans)
    assert fwd['sol']['data'].shape[1] == pos.shape[0]
    projs = list()
    for kind in ('meg', 'eeg'):
        kwargs = dict(meg=False, eeg=False, exclude=())
        kwargs.update({kind: True})
        picks = pick_types(info, **kwargs)
        if len(picks) > 0:
            ch_names = [info['ch_names'][pick] for pick in picks]
            projs.extend([
                Projection(
                    data=dict(data=p[np.newaxis, picks], row_names=None,
                              nrow=1, col_names=list(ch_names),
                              ncol=len(ch_names)),
                    kind=FIFF.FIFFV_PROJ_ITEM_DIP_FIX, explained_var=None,
                    active=False, desc=f'Dipole #{pi}')
                for pi, p in enumerate(fwd['sol']['data'].T, 1)])
    return projs


@verbose
def repeat_coreg(subject, subjects_dir=None, subjects_dir_old=None,
                 overwrite=False, verbose=None):
    """Repeat a mne coreg warping of an MRI.

    This is useful for example when bugs are fixed with
    :func:`mne.scale_mri`.

    Parameters
    ----------
    subject : str
        The subject name.
    subjects_dir : str | None
        The subjects directory where the redone subject should go.
        The template/surrogate MRI must also be in this directory.
    subjects_dir_old : str | None
        The subjects directory where the old subject is.
        Can be None to use ``subjects_dir``.
    overwrite : bool
        If True (default False), overwrite an existing subject directory
        if it exists.
    verbose : str | None
        The verbose level to use.

    Returns
    -------
    out_dir : str
        The output subject directory.
    """
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir)
    if subjects_dir_old is None:
        subjects_dir_old = subjects_dir
    config = mne.coreg.read_mri_cfg(subject, subjects_dir_old)
    n_params = config.pop('n_params')
    assert n_params in (3, 1), n_params
    out_dir = op.join(subjects_dir, subject)
    mne.coreg.scale_mri(subject_to=subject, subjects_dir=subjects_dir,
                        labels=False, annot=False, overwrite=overwrite,
                        **config)
    sol_file = op.join(subjects_dir, subject, 'bem',
                       '%s-5120-bem-sol.fif' % subject)
    if not op.isfile(sol_file):
        print('  Computing BEM solution')
        sol = mne.make_bem_solution(sol_file[:-8] + '.fif')
        mne.write_bem_solution(sol_file, sol)
    return out_dir


def convert_ANTS_surrogate(subject, trans, subjects_dir):
    """Convert an old ANTS surrogate to a modern one.

    Parameters
    ----------
    subject : str
        The subject name.
    trans : str
        The path to the subject's MRI<->head transformation.
    subjects_dir : str
        The subjects dir that contains the old ``subject`` MRI.

    Notes
    -----
    The "old" templates are the ones created by Eric Larson around 2019 and
    only include volumetric source spaces. The "modern" templates come from the
    2021 NeuroImage paper by O'Reilly et al. and use the same templates,
    just processed differently. Given a surrogate created using the old
    template, this function will create an equivalent one for the new
    template. It operates in-place by first backing up (renaming) the MRI
    directory for the subject, copying the ``-trans.fif`` file to that
    directory, and then creating the new surrogate and overwriting the old
    trans file.
    """
    # load morph params
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir)
    config = mne.coreg.read_mri_cfg(subject, subjects_dir)
    n_params = config.pop('n_params')
    subject_from = config['subject_from']
    if subject_from not in ('ANTS3-0Months3T', 'ANTS6-0Months3T',
                            'ANTS12-0Months3T'):
        raise RuntimeError('Cannot convert subject that used '
                           f'{repr(subject_from)} as a surrogate')
    age = int(subject_from.split('-')[0].split('S')[-1])
    assert n_params in (3, 1), n_params
    out_dir = op.join(subjects_dir, subject)
    backup_dir = op.join(subjects_dir, subject + '_old')
    if not isinstance(trans, (str, os.PathLike)):
        raise TypeError(f'trans must be path-like, got {type(trans)}')
    assert isinstance(trans, str)
    trans, trans_fname = mne.transforms._get_trans(trans, 'head', 'mri')
    if op.exists(backup_dir):
        raise RuntimeError(f'Backup dir {backup_dir} must not already exist')
    backup_trans = op.join(out_dir, op.basename(trans_fname))
    if op.exists(backup_trans):
        raise RuntimeError(f'Backup trans location {backup_trans} must not '
                           'exist')
    from_dir = op.join(subjects_dir, subject_from)
    if not op.isdir(from_dir):
        raise RuntimeError(f'Template MRI directory not found: {from_dir}')
    bem_path = op.join(
        from_dir, 'bem', f'{subject_from}-5120-5120-5120-bem-sol.fif')
    if not op.isfile(bem_path):
        raise RuntimeError(f'{subject_from} in {repr(subjects_dir)} does not '
                           'appear to be a new-style template, consider '
                           'running:\n\n'
                           'import shutil, mne\n'
                           f'shutil.rmtree({repr(from_dir)})\n'
                           f'mne.datasets.fetch_infant_template(\'{age}mo\''
                           f', subjects_dir={repr(subjects_dir)}'
                           ', verbose=True)\n')
    shutil.move(trans_fname, backup_trans)
    shutil.move(out_dir, backup_dir)
    print('Rescaling MRI (will be slow)...')
    mne.coreg.scale_mri(subject_to=subject, subjects_dir=subjects_dir,
                        labels=False, annot=False, overwrite=False,
                        **config)
    bem_path = op.join(
        out_dir, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
    sol = mne.make_bem_solution(mne.read_bem_surfaces(bem_path[:-8] + '.fif'))
    mne.write_bem_solution(bem_path, sol)
    # A factor beacuse Christian's MRIs weren't conformed:
    # tra = {
    #     3: [3, 8, 7.5], 6: [-1, 10.5, 10], 12: [0.5, 8, 15],
    # }
    # But these factors didn't completely explain the differences. So these
    # adjustments were done by eye, and confirmed by
    # surface-matching code that follows this function.
    tra = {
        3: [2, 7, 10], 6: [-1, 11, 8.5], 12: [-1, 10, 13],
    }
    x_rot = {3: -6.5, 6: 0, 12: 8}
    y_rot = {3: -2.5, 6: 2, 12: 0}
    rot = mne.transforms.rotation(x=np.deg2rad(x_rot[age]),
                                  y=np.deg2rad(y_rot[age]))
    tra = mne.transforms.translation(*tra[age])
    xform = rot @ tra
    xform[:3, 3] *= config['scale'] / 1000.  # scale and mm->m
    trans['trans'][:] = xform @ trans['trans']
    mne.transforms.write_trans(trans_fname, trans)


# This was used for the automatic fitting step:
"""
import numpy as np
from scipy.spatial import KDTree
import mne
from mne.transforms import apply_trans

for subject in ('ANTS3-0Months3T', 'ANTS6-0Months3T', 'ANTS12-0Months3T'):
    rr_from = mne.read_surface(f'/mnt/bakraid/larsoner/mri/Infants/subjects/{subject}/bem/outer_skin.surf')[0] / 1000.
    rr_to = mne.read_surface(f'/mnt/bakraid/data/structurals/{subject}/bem/outer_skin.surf')[0] / 1000.
    tree = KDTree(rr_to)
    transform = np.eye(4)
    for ii in range(10):
        rr_trans = apply_trans(transform, rr_from)
        dists, nearest = tree.query(rr_trans)
        use = np.arange(len(rr_from))
        a = rr_from[use]
        b = rr_to[nearest[use]]
        print(f'Iteration {ii}: {1000 * np.median(dists[use]):0.2f} mm')
        transform = mne.coreg.fit_matched_points(a, b)
        assert transform.shape == (4, 4)
    dists, nearest = tree.query(rr_trans)
    print(f'Done: {1000 * np.median(dists):0.2f} mm')
    with np.printoptions(precision=None, suppress=True, linewidth=150, floatmode='unique'):
        print(repr(transform))
"""  # noqa: E501
