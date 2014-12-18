# -*- coding: utf-8 -*-

import os
from os import path as op
import numpy as np
import time

from mne import (pick_types, pick_info, SourceEstimate, read_labels_from_annot,
                 Label, get_config, Epochs, compute_raw_data_covariance,
                 convert_forward_solution)
from mne.io import read_info, Raw
from mne.io.pick import _has_kit_refs
from mne.externals.six import string_types
from mne.forward.forward import _merge_meg_eeg_fwds
from mne.forward._make_forward import (_create_coils, _read_coil_defs,
                                       _compute_forwards, _to_forward_dict)
from mne.transforms import (read_trans, _get_mri_head_t_from_trans_file,
                            invert_transform, transform_surface_to)
from mne.source_space import (SourceSpaces, read_source_spaces,
                              _filter_source_spaces)
from mne.io.constants import FIFF
from mne.utils import logger, verbose
from mne.surface import read_bem_solution
from mne.simulation import generate_evoked


def _make_forward_solutions(info, mri, src, bem, dev_head_ts, mindist, n_jobs):
    """Calculate a forward solution for a subject

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info | str
        If str, then it should be a filename to a Raw, Epochs, or Evoked
        file with measurement information. If dict, should be an info
        dict (such as one from Raw, Epochs, or Evoked).
    mri : dict | str
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option).
    src : str | instance of SourceSpaces
        If string, should be a source space filename. Can also be an
        instance of loaded or generated SourceSpaces.
    bem : str
        Filename of the BEM (e.g., "sample-5120-5120-5120-bem-sol.fif") to
        use.
    dev_head_ts : list
        List of device<->head transforms.
    mindist : float
        Minimum distance of sources from inner skull surface (in mm).
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    fwd : generator
        A generator for each forward solution in dev_head_ts.

    Notes
    -----
    Some of the forward solution calculation options from the C code
    (e.g., `--grad`, `--fixed`) are not implemented here. For those,
    consider using the C command line tools or the Python wrapper
    `do_forward_solution`.
    """
    # Currently not (sup)ported:
    # 1. EEG Sphere model (not used much)
    # 2. --grad option (gradients of the field, not used much)
    # 3. --fixed option (can be computed post-hoc)
    # 4. --mricoord option (probably not necessary)

    # read the transformation from MRI to HEAD coordinates
    # (could also be HEAD to MRI)
    if isinstance(mri, string_types):
        if not op.isfile(mri):
            raise IOError('mri file "%s" not found' % mri)
        if op.splitext(mri)[1] in ['.fif', '.gz']:
            mri_head_t = read_trans(mri)
        else:
            mri_head_t = _get_mri_head_t_from_trans_file(mri)
    else:  # dict
        mri_head_t = mri
        mri = 'dict'

    if not isinstance(src, string_types):
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be a string or SourceSpaces')
    else:
        if not op.isfile(src):
            raise IOError('Source space file "%s" not found' % src)
    if not op.isfile(bem):
        raise IOError('BEM file "%s" not found' % bem)
    if not isinstance(info, (dict, string_types)):
        raise TypeError('info should be a dict or string')
    if isinstance(info, string_types):
        info_extra = op.split(info)[1]
        info = read_info(info, verbose=False)
    else:
        info_extra = 'info dict'

    # set default forward solution coordinate frame to HEAD
    # this could, in principle, be an option
    coord_frame = FIFF.FIFFV_COORD_HEAD

    # Report the setup
    logger.info('Setting up forward solutions')

    # Read the source locations
    if isinstance(src, string_types):
        src = read_source_spaces(src, verbose=False)
    else:
        # let's make a copy in case we modify something
        src = src.copy()
    nsource = sum(s['nuse'] for s in src)
    if nsource == 0:
        raise RuntimeError('No sources are active in these source spaces. '
                           '"do_all" option should be used.')
    logger.info('Read %d source spaces a total of %d active source locations'
                % (len(src), nsource))

    # it's actually usually a head->MRI transform, so we probably need to
    # invert it
    if mri_head_t['from'] == FIFF.FIFFV_COORD_HEAD:
        mri_head_t = invert_transform(mri_head_t)
    if not (mri_head_t['from'] == FIFF.FIFFV_COORD_MRI and
            mri_head_t['to'] == FIFF.FIFFV_COORD_HEAD):
        raise RuntimeError('Incorrect MRI transform provided')

    # make a new dict with the relevant information
    mri_id = dict(machid=np.zeros(2, np.int32), version=0, secs=0, usecs=0)
    info = dict(nchan=info['nchan'], chs=info['chs'], comps=info['comps'],
                ch_names=info['ch_names'],
                mri_file='', mri_id=mri_id, meas_file='',
                meas_id=None, working_dir=os.getcwd(),
                command_line='', bads=info['bads'])

    # MEG channels
    megnames = None
    picks = pick_types(info, meg=True, eeg=False, ref_meg=False,
                       exclude=[])
    nmeg = len(picks)
    if nmeg > 0:
        megchs = pick_info(info, picks)['chs']
        megnames = [info['ch_names'][p] for p in picks]
        logger.info('Read %3d MEG channels from %s'
                    % (len(picks), info_extra))

    # comp channels
    picks = pick_types(info, meg=False, ref_meg=True, exclude=[])
    ncomp = len(picks)
    if (ncomp > 0):
        compchs = pick_info(info, picks)['chs']
        logger.info('Read %3d MEG compensation channels from %s'
                    % (ncomp, info_extra))
        # We need to check to make sure these are NOT KIT refs
        if _has_kit_refs(info, picks):
            err = ('Cannot create forward solution with KIT '
                   'reference channels. Consider using '
                   '"ignore_ref=True" in calculation')
            raise NotImplementedError(err)
    ncomp_data = len(info['comps'])
    ref_meg = True
    picks = pick_types(info, meg=True, ref_meg=ref_meg, exclude=[])
    meg_info = pick_info(info, picks)

    # EEG channels
    eegnames = None
    picks = pick_types(info, meg=False, eeg=True, ref_meg=False,
                       exclude=[])
    neeg = len(picks)
    if neeg > 0:
        eegchs = pick_info(info, picks)['chs']
        eegnames = [info['ch_names'][p] for p in picks]
        logger.info('Read %3d EEG channels from %s'
                    % (len(picks), info_extra))

    if neeg <= 0 and nmeg <= 0:
        raise RuntimeError('Could not find any MEG or EEG channels')

    # Create coil descriptions with transformation to head or MRI frame
    templates = _read_coil_defs(verbose=False)
    if nmeg > 0 and ncomp > 0:  # Compensation channel information
        logger.info('%d compensation data sets in %s'
                    % (ncomp_data, info_extra))

    # Transform the source spaces into the appropriate coordinates
    # (will either be HEAD or MRI)
    for s in src:
        transform_surface_to(s, coord_frame, mri_head_t)

    # Prepare the BEM model
    bem = read_bem_solution(bem, verbose=False)
    if neeg > 0 and len(bem['surfs']) == 1:
        raise RuntimeError('Cannot use a homogeneous model in EEG '
                           'calculations')
    # fwd_bem_set_head_mri_t: Set the coordinate transformation
    to, fro = mri_head_t['to'], mri_head_t['from']
    if fro == FIFF.FIFFV_COORD_HEAD and to == FIFF.FIFFV_COORD_MRI:
        bem['head_mri_t'] = mri_head_t
    elif fro == FIFF.FIFFV_COORD_MRI and to == FIFF.FIFFV_COORD_HEAD:
        bem['head_mri_t'] = invert_transform(mri_head_t)
    else:
        raise RuntimeError('Improper coordinate transform')

    # Circumvent numerical problems by excluding points too close to the skull
    idx = np.where(np.array([s['id'] for s in bem['surfs']])
                   == FIFF.FIFFV_BEM_SURF_ID_BRAIN)[0]
    if len(idx) != 1:
        raise RuntimeError('BEM model does not have the inner skull '
                           'triangulation')
    logger.info('Filtering source spaces')
    _filter_source_spaces(bem['surfs'][idx[0]], mindist, mri_head_t, src,
                          n_jobs, verbose=False)

    picks = pick_types(info, meg=True, eeg=True, ref_meg=False, exclude=[])
    info = pick_info(info, picks)
    source_rr = np.concatenate([s['rr'][s['vertno']] for s in src])
    # deal with free orientations:
    for key in ['working_dir', 'command_line']:
        if key in src.info:
            del src.info[key]

    # Time to do the heavy lifting: EEG first, then MEG
    eegfwd = None
    if neeg > 0:
        eegels, _ = _create_coils(eegchs, coil_type='eeg', coilset=templates)
        eegfwd = _compute_forwards(src, bem, [eegels], [None], [None], [None],
                                   [None], ['eeg'], n_jobs, verbose=False)[0]
    eegfwd = _to_forward_dict(eegfwd, None, eegnames, coord_frame,
                              FIFF.FIFFV_MNE_FREE_ORI)

    megcoils, megcf, compcoils, compcf = None, None, None, None
    assert nmeg > 0  # otherwise this function is pointless!
    durations = list()
    for ti, dev_head_t in enumerate(dev_head_ts):
        # could be *slightly* more efficient not to do this N times,
        # but the cost here is tiny compared to actual fwd calculation
        logger.info('Computing gain matrix for transform #%s/%s'
                    % (ti + 1, len(dev_head_ts)))
        t0 = time.time()
        megcoils = _create_coils(megchs, FIFF.FWD_COIL_ACCURACY_ACCURATE,
                                 dev_head_t, coil_type='meg',
                                 coilset=templates)[0]
        megcf = megcoils[0]['coord_frame']
        if ncomp > 0:
            compcoils = _create_coils(compchs,
                                      FIFF.FWD_COIL_ACCURACY_NORMAL,
                                      dev_head_t, coil_type='meg',
                                      coilset=templates)[0]
            compcf = compcoils[0]['coord_frame']
        megfwd = _compute_forwards(src, bem, [megcoils], [megcf], [compcoils],
                                   [compcf], [meg_info], ['meg'], n_jobs,
                                   verbose=False)[0]
        megfwd = _to_forward_dict(megfwd, None, megnames, coord_frame,
                                  FIFF.FIFFV_MNE_FREE_ORI)
        fwd = _merge_meg_eeg_fwds(megfwd, eegfwd, verbose=False)

        # pick out final dict info
        nsource = fwd['sol']['data'].shape[1] // 3
        source_nn = np.tile(np.eye(3), (nsource, 1))
        fwd.update(dict(nchan=fwd['sol']['data'].shape[0], nsource=nsource,
                        info=info, src=src, source_nn=source_nn,
                        source_rr=source_rr, surf_ori=False,
                        mri_head_t=mri_head_t))
        fwd['info']['mri_head_t'] = mri_head_t
        fwd['info']['dev_head_t'] = dev_head_t
        durations.append(time.time() - t0)
        yield fwd, durations


@verbose
def simulate_movement(raw, stc, trans, src, bem, snr=0., cov=None, eeg=True,
                      add_stationary=False, n_jobs=1, verbose=True):
    """Simulate raw data with head movements

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to use. The measurement info, including the
        head positions, will be used to simulate data.
    stc : instance of SourceEstimate
        The source estimate to use to simulate data.
    trans : dict | str
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option).
    src : str | instance of SourceSpaces
        If string, should be a source space filename. Can also be an
        instance of loaded or generated SourceSpaces.
    bem : str
        Filename of the BEM (e.g., "sample-5120-5120-5120-bem-sol.fif").
    snr : float
        SNR of the simulated data.
    cov : instance of Covariance
        The sensor covariance matrix used to generate noise.
    eeg : bool
        Toggle EEG data simulation.
    add_stationary : bool | str
        If True, construct a second Raw instance that is stationary,
        and uses either the first head position (True) or the mediean
        head position ('median') for the duration of the raw file.
    n_jobs : int
        Number of jobs to use.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of Raw
        The simulated raw file.
    raw_control : instance of Raw
        The simulated raw file with no movement, only returned if
        ``add_stationary`` is not False.

    TODO:
    - Add spatial jitter of HPI?
    - Add on-the-fly plotting
    - Add control condition (using median inputted head position)
    """
    if isinstance(raw, string_types):
        raw = Raw(raw, preload=True)
    if not raw.preload:
        raise RuntimeError('raw file must be preloaded')
    info = raw.info

    dev_head_ts = [info['dev_head_t']] * 2

    if not isinstance(info, (dict, string_types)):
        raise TypeError('info should be a dict or string')
    if isinstance(info, string_types):
        info = read_info(info, verbose=False)
    picks = pick_types(info, meg=True, eeg=eeg)  # for simulation
    fwd_info = pick_info(raw.info, picks)
    assert not add_stationary  # XXX Not implemented yet

    # Create a dummy evoked object
    n_trans = len(dev_head_ts)
    events = np.array([[raw.first_samp, 0, 1]])
    epochs = Epochs(raw, events, 1, 0, stc.times[-1] - stc.times[0],
                    picks=picks, preload=True, reject=None, flat=None,
                    verbose=False)
    assert len(epochs) == 1
    evoked = epochs.average()
    assert len(evoked.times) == len(stc.times)

    # Create a covariance if none was supplied
    if cov is None:
        logger.info('Comuting raw data covariance for noise simulation')
        cov = compute_raw_data_covariance(raw, verbose=False)

    # Create our data buffer    
    data_buffer = np.zeros((len(picks), 0))
    raw_offset = 0
    # XXX set up all time indices ahead of time
    for fi, (fwd, durs) in enumerate(_make_forward_solutions(
            fwd_info, trans, src, bem, dev_head_ts, 1.0, n_jobs)):
        fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                       verbose=False)
        logger.info('  Simulating data using gain matrix')

        # simulate data
        out = generate_evoked(fwd, stc, evoked, cov, snr, verbose=False)
        #use_len = 0
        #while data_buffer.shape[1] < use_len:
        #    data_buffer = np.concatenate((data_buffer, out.data), axis=1)
        #raw._data[picks, raw_offset:raw_offset+use_len] = data_buffer[:, :use_len]
        #data_buffer = data_buffer[:, use_len:]

        # give a status update
        n_trans - fi - 1
        time_str = time.strftime('%H:%M:%S', time.gmtime(
            np.median(durs) * (n_trans - fi - 1)))
        if fi < n_trans - 1:
            logger.info('  Estimated time remaining: %s' % time_str)
    return raw


if __name__ == '__main__':
    print('Setting up data')
    subjects_dir = get_config('SUBJECTS_DIR')
    subj, subject = 'eric_voc_007', 'AKCLEE_107'
    file_dir = '/home/larsoner/Documents/python/larsoner/voc_meg/%s/' % subj
    fname_raw = op.join(file_dir, 'raw_fif', '%s_01_raw.fif' % subj)
    trans = file_dir + 'trans/%s-trans.fif' % subj
    bem = op.join(subjects_dir, subject, 'bem',
                  '%s-5120-5120-5120-bem-sol.fif' % subject)
    src = read_source_spaces(op.join(subjects_dir, subject, 'bem',
                             '%s-oct-6-src.fif' % subject))
    raw = Raw(fname_raw, allow_maxshield=True, preload=True)

    # construct appropriate STC
    dur = 1.
    vertices = [s['vertno'] for s in src]
    n_vertices = sum(s['nuse'] for s in src)
    data = np.ones((n_vertices, int(dur * raw.info['sfreq'])))
    stc = SourceEstimate(data, vertices, -0.2, 1. / raw.info['sfreq'], subject)

    # limit activation to two vertices
    labels = []
    for hi, hemi in enumerate(('lh', 'rh')):
        label = read_labels_from_annot(subject, 'aparc.a2009s', hemi,
                                       regexp='G_temp_sup-G_T_transv')[0]
        center = stc.in_label(label).center_of_mass(restrict_vertices=True)
        assert center[1] == hi
        labels.append(Label([center[0]], hemi=hemi))
    stc = stc.in_label(labels[0] + labels[1])
    stc.data.fill(0)
    stc.data[:, np.where(stc.times > 0)[0][0]] = 1e-9

    raw = simulate_movement(raw, stc, trans, src, bem, n_jobs=6)
