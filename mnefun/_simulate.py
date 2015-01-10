# -*- coding: utf-8 -*-

import os
from os import path as op
import numpy as np
import time
import warnings
from copy import deepcopy

from mne import (pick_types, pick_info, SourceEstimate, pick_channels,
                 compute_raw_data_covariance, convert_forward_solution,
                 get_chpi_positions, EvokedArray)
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
from mne.utils import logger, verbose, check_random_state
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
    for ti, dev_head_t in enumerate(dev_head_ts):
        # could be *slightly* more efficient not to do this N times,
        # but the cost here is tiny compared to actual fwd calculation
        logger.info('Computing gain matrix for transform #%s/%s'
                    % (ti + 1, len(dev_head_ts)))
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
        yield fwd


def _restrict_source_space_to(src, vertices):
    """Helper to trim down a source space"""
    assert len(src) == len(vertices)
    src = deepcopy(src)
    for s, v in zip(src, vertices):
        s['inuse'].fill(0)
        s['nuse'] = len(v)
        s['vertno'] = v
        s['inuse'][s['vertno']] = 1
        del s['pinfo']
        del s['nuse_tri']
        del s['use_tris']
        del s['patch_inds']
    return src


@verbose
def simulate_movement(raw, pos, stc, trans, src, bem, snr=0., cov=None,
                      snr_tmin=None, snr_tmax=None, mindist=1.0,
                      random_state=None, n_jobs=1, verbose=True):
    """Simulate raw data with head movements

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to use. The measurement info, including the
        head positions, will be used to simulate data.
    pos : str | None
        Name of the position estimates file. Should be in the format of
        the files produced by maxfilter-produced. If None, a fixed
        head position (using ``raw.info['dev_head_t']``) will be used.
    stc : instance of SourceEstimate
        The source estimate to use to simulate data. Must have the same
        sample rate as the raw data.
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
    cov : instance of Covariance | None
        The sensor covariance matrix used to generate noise. If None,
        the covariance will be estimated from the raw data.
    snr_tmin : float | None
        Minimum time to use in SNR computations. None will use the starting
        time.
    snr_tmax : float
        Maximum time to use in SNR computations. None will use the ending
        time.
    mindist : float
        Minimum distance between sources and the inner skull boundary
        to use during forward calculation.
    random_state : None | int | np.random.RandomState
        To specify the random generator state.
    n_jobs : int
        Number of jobs to use.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of Raw
        The simulated raw file.

    Notes
    -----
    Events coded with number 1 will be placed in the raw files in the
    trigger channel STI101 at the t=0 times of the SourceEstimates.

    Remaining issues:

        * Clean data a little bit?
        * Should projections be disabled?
        * How to add CHPI signals back in? Band-pass and add!?!
    """
    if isinstance(raw, string_types):
        with warnings.catch_warnings(record=True):
            raw = Raw(raw, allow_maxshield=True, verbose=False)
    else:
        raw = raw.copy()

    if not isinstance(stc, SourceEstimate):
        raise TypeError('stc must be a SourceEstimate')
    if not np.allclose(raw.info['sfreq'], 1. / stc.tstep):
        raise ValueError('stc and raw must have same sample rate')
    rng = check_random_state(random_state)

    if pos is None:  # use pos from file
        dev_head_ts = [raw.info['dev_head_t']]
        offsets = np.array([0, raw.n_times])
    else:
        transs, rots, ts = get_chpi_positions(pos, verbose=False)
        dev_head_ts = [np.r_[np.c_[r, t[:, np.newaxis]], [[0, 0, 0, 1]]]
                       for r, t in zip(rots, transs)]
        if not (ts >= 0).all():  # pathological if not
            raise RuntimeError('Cannot have t < 0 in transform file')
        t0 = raw.first_samp / raw.info['sfreq']
        if t0 < ts[0]:
            ts = np.r_[[t0], ts]
            dev_head_ts.insert(0, raw.info['dev_head_t']['trans'])
        dev_head_ts = [dict(trans=d, to=raw.info['dev_head_t']['to'])
                       for d in dev_head_ts]
        ts -= ts[0]  # re-reference
        offsets = np.r_[raw.time_as_index(ts), raw.n_times]
        del transs, rots, ts
    assert np.array_equal(offsets, np.unique(offsets))
    assert len(offsets) == len(dev_head_ts) + 1

    picks = pick_types(raw.info, meg=True, eeg=True)  # for simulation
    fwd_info = pick_info(raw.info, picks)
    logger.info('Setting up raw data simulation using %s head position%s'
                % (len(dev_head_ts), 's' if len(dev_head_ts) != 1 else ''))

    # Create a covariance if none was supplied
    raw.preload_data()
    if cov is None:
        logger.info('Computing raw data covariance for noise simulation')
        cov = compute_raw_data_covariance(raw, verbose=False)
    src = _restrict_source_space_to(src, stc.vertices)

    evoked = EvokedArray(np.zeros((len(picks), len(stc.times))), fwd_info,
                         stc.tmin, verbose=False)
    stc_event_idx = np.argmin(np.abs(stc.times))
    event_ch = pick_channels(raw.info['ch_names'], ['STI101'])[0]
    simulated = np.zeros(raw.n_times, bool)
    stc_indices = np.arange(raw.n_times) % len(stc.times)
    t0 = time.time()
    raw._data[event_ch, ].fill(0)
    for fi, fwd in enumerate(_make_forward_solutions(
            fwd_info, trans, src, bem, dev_head_ts, mindist, n_jobs)):
        fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                       verbose=False)

        time_slice = slice(offsets[fi], offsets[fi+1])
        assert not simulated[time_slice].any()
        stc_idxs = stc_indices[time_slice]
        event_idxs = np.where(stc_idxs == stc_event_idx)[0] + offsets[fi]
        simulated[time_slice] = True
        logger.info('  Simulating data for %0.1f-%0.1f sec with %s events'
                    % (tuple(offsets[fi:fi+2] / raw.info['sfreq'])
                       + (len(event_idxs),)))

        # simulate data
        sim_stc = SourceEstimate(stc.data[:, stc_idxs], stc.vertices,
                                 stc.tmin, stc.tstep)
        out = generate_evoked(fwd, sim_stc, evoked, cov, snr,
                              tmin=snr_tmin, tmax=snr_tmax,
                              random_state=rng, verbose=False)
        assert out.data.shape[0] == len(picks)
        assert out.data.shape[1] == len(stc_idxs)
        raw._data[picks, time_slice] = out.data
        raw._data[event_ch, event_idxs] = 1.
    assert simulated.all()
    logger.info('Done')
    return raw
