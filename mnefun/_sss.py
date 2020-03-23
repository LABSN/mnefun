"""Head position estimation, Maxwell filtering, annotations."""

import os
import os.path as op
import time
import shutil
import subprocess
import sys
import warnings

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.spatial.distance import cdist

from mne import (read_trans, Transform, read_annotations, Annotations,
                 pick_types)
from mne.bem import fit_sphere_to_headshape
from mne.chpi import read_head_pos, write_head_pos, filter_chpi
from mne.externals.h5io import read_hdf5, write_hdf5
from mne.io import read_raw_fif, BaseRaw, write_info, read_info
from mne.preprocessing import maxwell_filter
from mne.transforms import (quat_to_rot, rot_to_quat, invert_transform,
                            apply_trans)
from mne.utils import (_TempDir, run_subprocess, _pl, verbose, logger,
                       use_log_level)


from ._paths import get_raw_fnames, _prebad, _get_config_file
from ._utils import get_args

_data_dir = op.join(op.dirname(__file__), 'data')


def run_sss(p, subjects, run_indices):
    """Run SSS preprocessing remotely (only designed for *nix platforms) or
    locally using Maxwell filtering in mne-python"""
    from ._sss_legacy import _check_trans_file
    if p.sss_type == 'python':
        print('  Applying SSS locally using mne-python')
        run_sss_locally(p, subjects, run_indices)
    else:
        for si, subj in enumerate(subjects):
            files = get_raw_fnames(p, subj, 'raw', False, True,
                                   run_indices[si])
            n_files = len(files)
            files = ':'.join([op.basename(f) for f in files])
            erm = get_raw_fnames(p, subj, 'raw', 'only', True, run_indices[si])
            n_files += len(erm)
            erm = ':'.join([op.basename(f) for f in erm])
            erm = ' --erm ' + erm if len(erm) > 0 else ''
            assert isinstance(p.tsss_dur, float) and p.tsss_dur > 0
            st = ' --st %s' % p.tsss_dur
            if p.sss_format not in ('short', 'long', 'float'):
                raise RuntimeError('format must be short, long, or float')
            fmt = ' --format ' + p.sss_format
            assert p.movecomp in ['inter', None]
            mc = ' --mc %s' % str(p.movecomp).lower()
            _check_trans_file(p)
            trans = ' --trans ' + p.trans_to
            run_sss = (op.join(p.sws_dir, 'run_sss.sh') + st + fmt + trans +
                       ' --subject ' + subj + ' --files ' + files + erm + mc +
                       ' --args=\"%s\"' % p.mf_args)
            cmd = ['ssh', '-p', str(p.sws_port), p.sws_ssh, run_sss]
            s = 'Remote output for %s on %s files:' % (subj, n_files)
            print('-' * len(s))
            print(s)
            print('-' * len(s))
            run_subprocess(cmd, stdout=sys.stdout, stderr=sys.stderr)
            print('-' * 70, end='\n\n')


@verbose
def run_sss_command(fname_in, options, fname_out, host='kasga', port=22,
                    fname_pos=None, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, prefix='', work_dir='~/',
                    throw_error=True, verbose=None):
    """Run Maxfilter remotely and fetch resulting file.

    Parameters
    ----------
    fname_in : str
        The filename to process.
    options : str
        The command-line options for Maxfilter.
    fname_out : str | None
        Output filename to use to store the result on the local machine.
        None will output to a temporary file.
    host : str
        The SSH/scp host to run the command on.
    fname_pos : str | None
        The ``-hp fname_pos`` to use with MaxFilter.
    stdout : file-like | None
        Where to send stdout.
    stderr : file-like | None
        Where to send stderr.
    prefix : str
        The text to prefix to messages.
    work_dir : str
        Where to store the temporary files.
    throw_error : bool
        If True, throw an error if the command fails. If False,
        return anyway, including the code.

    Returns
    -------
    stdout : str
        The standard output of the ``maxfilter`` call.
    stderr : str
        The standard error of the ``maxfilter`` call.
    code : int
        Only returned if throw_error=False.
    """
    if not isinstance(host, str):
        raise ValueError('host must be a string, got %r' % (host,))
    # let's make sure we can actually write where we want
    if isinstance(fname_in, str) and not op.isfile(fname_in):
        raise IOError('input file not found: %s' % fname_in)
    if fname_out is not None:
        if not op.isdir(op.dirname(op.abspath(fname_out))):
            raise IOError('output directory for output file does not exist')
    if any(x in options for x in ('-f ', '-o ', '-hp ')):
        raise ValueError('options cannot contain -o, -f, or -hp, these are '
                         'set automatically')
    port = str(int(port))
    t0 = time.time()
    remote_in = op.join(work_dir, 'temp_%s_raw.fif' % t0)
    remote_out = op.join(work_dir, 'temp_%s_raw_sss.fif' % t0)
    remote_pos = op.join(work_dir, 'temp_%s_raw_sss.pos' % t0)
    print('%sOn %s: ' % (prefix, host), end='')
    if isinstance(fname_in, str):
        fname_in = op.realpath(fname_in)  # in case it's a symlink
    elif fname_in is not None:
        assert isinstance(fname_in, BaseRaw)
        tempdir = _TempDir()
        temp_fname = op.join(tempdir, 'temp_raw.fif')
        fname_in.save(temp_fname)
        fname_in = temp_fname
        del temp_fname
    if fname_in is not None:
        print('copying, ', end='')
        try:
            _push_remote(fname_in, host, port, remote_in)
        finally:
            shutil.rmtree(tempdir)
        in_out = '-f ' + remote_in + ' -o ' + remote_out + ' '
    else:
        in_out = ''

    if fname_pos is not None:
        options += ' -hp ' + remote_pos

    print('maxfilter %s' % (options,), end='')
    cmd = ['ssh', '-p', port, host,
           'maxfilter ' + in_out + options]
    output = run_subprocess(
        cmd, return_code=True, stdout=stdout, stderr=stderr)
    output, code = output[:2], output[2]
    # pull files if things were good
    if code == 0:
        if fname_out is not None:
            print(', copying to %s' % (op.basename(fname_out),), end='')
        if fname_pos is not None:
            try:
                _pull_remote(host, port, remote_pos, fname_pos)
            except Exception:
                pass
    # always clean up
    files = []
    files += [remote_in, remote_out] if fname_in is not None else []
    files += [remote_pos] if fname_pos is not None else []
    if files:
        print(', cleaning', end='')
        clean_cmd = ['ssh', '-p', port, host, 'rm -f ' + ' '.join(files)]
        try:
            run_subprocess(
                clean_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass
    # now throw an error
    if code != 0:
        if 'maxfilter: command not' in output[1]:
            print(output)
            raise RuntimeError(
                '\nMaxFilter could not be run on the remote machine, '
                'consider adding the following line to your ~/.bashrc on '
                'the remote machine:\n\n'
                'export PATH=${PATH}:/neuro/bin/util:/neuro/bin/X11\n')
        if throw_error:
            print(output)
            raise subprocess.CalledProcessError(code, cmd)
    print(' (%i sec)' % (time.time() - t0,))
    if not throw_error:
        output = output + (code,)
    return output


@verbose
def run_sss_positions(fname_in, fname_out, host='kasga', opts='-force',
                      port=22, prefix='  ', work_dir='~/', t_window=None,
                      t_step_min=None, dist_limit=None, gof_limit=0.98,
                      verbose='error'):
    """Run Maxfilter remotely and fetch resulting file

    Parameters
    ----------
    fname_in : str
        The filename to process. Additional ``-1`` files will be
        automatically detected.
    fname_out : str
        Output filename to use to store the resulting head positions
        on the local machine.
    host : str
        The SSH/scp host to run the command on
    opts : str
        Additional command-line options to pass to MaxFilter.
    port : int
        The SSH port.
    prefix : str
        The prefix to use when printing status updates.
    work_dir : str
        Where to store the temporary files.
    t_window : float | None
        Time window (sec) to use.
    dist_limit : float | None
        Distance limit (m) to use.
    verbose : str
        MNE verbose level, effectively controls whether or not the
        stdout/stderr is printed while running.
    """
    # let's make sure we can actually write where we want
    if not op.isfile(fname_in):
        raise IOError('input file not found: %s' % fname_in)
    if not op.isdir(op.dirname(op.abspath(fname_out))):
        raise IOError('output directory for output file does not exist')
    pout = op.dirname(fname_in)
    fnames_in = [fname_in]
    for ii in range(1, 11):
        next_name = op.splitext(fname_in)[0] + '-%s' % ii + '.fif'
        if op.isfile(next_name):
            fnames_in.append(next_name)
        else:
            break
    if t_window is not None:
        opts += ' -hpiwin %d' % (round(1000 * t_window),)
    if t_step_min is not None:
        opts += ' -hpistep %d' % (round(1000 * t_step_min),)
    if dist_limit is not None:
        opts += ' -hpie %d' % (round(1000 * dist_limit),)
    else:
        dist_limit = 0.005
    gof_limit = float(gof_limit)
    opts += ' -hpig %f' % (gof_limit,)

    t0 = time.time()
    print('%sOn %s: copying' % (prefix, host), end='')
    cmd = ['rsync', '--partial', '-Lave', 'ssh -p %s' % port,
           '--include', '*/']
    for fname in fnames_in:
        cmd += ['--include', op.basename(fname)]
    cmd += ['--exclude', '*', op.dirname(fnames_in[0]) + '/',
            '%s:%s' % (host, work_dir)]
    run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    remote_ins = [op.join(work_dir, op.basename(f)) for f in fnames_in]
    fnames_out = [op.basename(r)[:-4] + '.pos' for r in remote_ins]
    for fi, file_out in enumerate(fnames_out):
        remote_out = op.join(work_dir, 'temp_%s_raw_quat.fif' % t0)
        remote_hp = op.join(work_dir, 'temp_%s_hp.txt' % t0)

        print(', running -headpos %s' % opts, end='')
        cmd = ['ssh', '-p', str(port), host,
               '/neuro/bin/util/maxfilter -f ' + remote_ins[fi] + ' -o ' +
               remote_out +
               ' -headpos -format short -hp ' + remote_hp + ' ' + opts]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(', copying', end='')
        _pull_remote(host, port, remote_hp, op.join(pout, file_out))
        cmd = ['ssh', '-p', str(port), host, 'rm -f %s %s %s'
               % (remote_ins[fi], remote_hp, remote_out)]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # concatenate hp pos file for split raw files if any
    data = []
    next_pre = ', '
    gof_limit = 0.98
    for f in fnames_out:
        this_pos = read_head_pos(op.join(pout, f))
        # sanitize the stupid thing; sometimes MaxFilter produces a line of all
        # (or mostly) zeros, presumably because on the first sample it can say
        # "fit not good enough, use the last one" but the last one is just
        # whatever is in the malloc()'ed array.
        # Let's use a heuristic of at least two entries being
        # zero, or any invalid quat, or an invalid position, and take the
        # next valid one
        valid = _pos_valid(this_pos, dist_limit, gof_limit)
        if len(this_pos > 0) and not valid[0]:
            first_valid = np.where(valid)[0][0]
            repl = this_pos[first_valid].copy()
            repl[7:9] = [gof_limit, dist_limit]
            print('\n%s... found %d invalid position%s from MaxFilter, '
                  'replacing with: %r'
                  % (prefix, first_valid, _pl(first_valid),
                     repl[1:].tolist()), end='')
            this_pos[:first_valid, 1:] = repl[1:]
            next_pre = '\n%s... ' % (prefix,)
        data.append(this_pos)
        os.remove(op.join(pout, f))
    pos_data = np.concatenate(np.array(data))
    print(next_pre + 'writing', end='')
    write_head_pos(fname_out, pos_data)
    print(' (%i sec)' % (time.time() - t0,))


def _read_raw_prebad(p, subj, fname, disp=True, prefix=' ' * 6):
    """Read SmartShield raw instance and add bads."""
    prebad_file = _prebad(p, subj)
    raw = read_raw_fif(fname, allow_maxshield='yes')
    raw.fix_mag_coil_types()
    # First load our manually marked ones (if present)
    _load_meg_bads(raw, prebad_file, disp=disp, prefix=prefix)
    # Next run Maxfilter for automatic bad channel selection
    if p.mf_autobad:
        maxbad_file = op.splitext(fname)[0] + '_maxbad.txt'
        _maxbad(p, raw, maxbad_file)
        _load_meg_bads(raw, maxbad_file, disp=disp, prefix=prefix, append=True)
    return raw


def _get_origin(p, raw):
    if p.sss_origin == 'auto':
        R, origin = fit_sphere_to_headshape(
            raw.info, verbose=False, units='m')[:2]
        kind, extra = 'automatic', ' R=%0.1f' % (1000 * R,)
    else:
        kind, extra, origin = 'manual', '', p.sss_origin
    return origin, kind, extra


def _get_cal_ct_file(p):
    if getattr(p, 'cal_file', 'uw') == 'uw':
        cal_file = op.join(_data_dir, 'sss_cal.dat')
    else:
        cal_file = p.cal_file
    if getattr(p, 'ct_file', 'uw') == 'uw':
        ct_file = op.join(_data_dir, 'ct_sparse.fif')
    else:
        ct_file = p.ct_file
    return cal_file, ct_file


def run_sss_locally(p, subjects, run_indices):
    """Run SSS locally using maxwell filter in python

    See Also
    --------
    mne.preprocessing.maxwell_filter
    """
    from mne.annotations import _handle_meas_date
    cal_file, ct_file = _get_cal_ct_file(p)
    assert isinstance(p.tsss_dur, float) and p.tsss_dur > 0
    st_duration = p.tsss_dur
    assert (isinstance(p.sss_regularize, str) or
            p.sss_regularize is None)
    reg = p.sss_regularize

    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('    Maxwell filtering subject %g/%g (%s).'
                  % (si + 1, len(subjects), subj))
        # locate raw files with splits
        sss_dir = op.join(p.work_dir, subj, p.sss_dir)
        if not op.isdir(sss_dir):
            os.mkdir(sss_dir)
        raw_files = get_raw_fnames(p, subj, 'raw', erm=False,
                                   run_indices=run_indices[si])
        raw_files_out = get_raw_fnames(p, subj, 'sss', erm=False,
                                       run_indices=run_indices[si])
        erm_files = get_raw_fnames(p, subj, 'raw', 'only')
        erm_files_out = get_raw_fnames(p, subj, 'sss', 'only')

        # get the destination head position
        assert isinstance(p.trans_to, (str, tuple, type(None)))
        trans_to = _load_trans_to(p, subj, run_indices[si])
        #  process raw files
        raw_head_pos = raw_annot = raw_info = None
        assert len(raw_files) > 0
        for ii, (r, o) in enumerate(zip(raw_files + erm_files,
                                        raw_files_out + erm_files_out)):
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')
            raw = _read_raw_prebad(p, subj, r, disp=ii == 0).load_data()
            if ii == 0:
                origin, kind, extra = _get_origin(p, raw)
                print('      Using %s origin=[%0.1f, %0.1f, %0.1f]%s mm' %
                      ((kind,) + tuple(1000 * np.array(origin, float)) +
                       (extra,)))
                del kind, extra
            print('      Processing %s ...' % op.basename(r))

            # For the empty room files, mimic the necessary components
            if r in erm_files:
                for key in ('dev_head_t', 'hpi_meas', 'hpi_subsystem', 'dig'):
                    raw.info[key] = raw_info[key]
                if raw_head_pos is not None:
                    raw_head_pos[:, 0] -= raw_head_pos[0, 0]
                    raw_head_pos[:, 0] += raw.first_samp / raw.info['sfreq']
                if raw_annot is not None:
                    meas_date = _handle_meas_date(raw.info['meas_date'])
                    try:
                        raw_annot.orig_time = meas_date
                    except AttributeError:
                        raw_annot._orig_time = meas_date
                head_pos, annot = raw_head_pos, raw_annot
            else:
                # estimate head position for movement compensation
                head_pos, annot, _ = _head_pos_annot(
                    p, subj, r, prefix='        ')
                if raw_info is None:
                    if head_pos is None:
                        raw_head_pos = None
                    else:
                        raw_head_pos = head_pos.copy()
                    raw_annot = annot
                    raw_info = raw.info.copy()

            emit_warning = r not in erm_files
            try:
                raw.set_annotations(annot, emit_warning=emit_warning)
            except AttributeError:
                raw.annotations = annot

            # filter cHPI signals
            if p.filter_chpi:
                t0 = time.time()
                print('        Filtering cHPI signals ... ', end='')
                raw = filter_chpi(
                    raw, t_window=_get_t_window(p, raw), verbose=False)
                print('%i sec' % (time.time() - t0,))

            # apply maxwell filter
            t0 = time.time()
            print('        Running maxwell_filter ...', end='')
            raw_sss = maxwell_filter(
                raw, head_pos=head_pos,
                origin=origin, int_order=p.int_order,
                ext_order=p.ext_order, calibration=cal_file,
                cross_talk=ct_file, st_correlation=p.st_correlation,
                st_duration=st_duration, destination=trans_to,
                coord_frame='head', regularize=reg, bad_condition='warning')
            print('%i sec' % (time.time() - t0,))
            raw_sss.save(o, overwrite=True, buffer_size_sec=None)


def _load_trans_to(p, subj=None, run_indices=None, raw=None):
    from mne.transforms import _ensure_trans
    if getattr(p, 'trans_to', None) is None:
        trans_to = None if raw is None else raw.info['dev_head_t']
    elif isinstance(p.trans_to, str):
        if p.trans_to == 'median':
            trans_to = op.join(p.work_dir, subj, p.raw_dir,
                               subj + '_median_pos.fif')
            if not op.isfile(trans_to):
                calc_median_hp(p, subj, trans_to, run_indices)
        elif p.trans_to == 'twa':
            trans_to = op.join(p.work_dir, subj, p.raw_dir,
                               subj + '_twa_pos.fif')
            if not op.isfile(trans_to):
                calc_twa_hp(p, subj, trans_to, run_indices)
        else:
            trans_to = p.trans_to
        trans_to = read_trans(trans_to)
    else:
        trans_to = np.array(p.trans_to, float)
        t = np.eye(4)
        if trans_to.shape == (4,):
            theta = np.deg2rad(trans_to[3])
            t[1:3, 1:3] = [[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]]
        elif trans_to.shape != (3,):
            raise ValueError('trans_to must have 3 or 4 elements, '
                             'got shape %s' % (trans_to.shape,))
        t[:3, 3] = trans_to[:3]
        trans_to = Transform('meg', 'head', t)
    if trans_to is not None:
        trans_to = _ensure_trans(trans_to, 'meg', 'head')
    return trans_to


def _push_remote(local, host, port, remote):
    cmd = ['rsync', '--partial', '--rsh', 'ssh -p %s' % port,
           local, host + ':' + remote]
    return run_subprocess(cmd)


def _pull_remote(host, port, remote, local):
    cmd = ['rsync', '--partial', '--rsh', 'ssh -p %s' % port,
           host + ':' + remote, local]
    return run_subprocess(cmd)


def _load_meg_bads(raw, prebad_file, disp=True, prefix='     ',
                   append=False):
    """Helper to load MEG bad channels from a file (pre-MF)"""
    with open(prebad_file, 'r') as fid:
        lines = fid.readlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    if len(lines) > 0:
        try:
            int(lines[0][0])
        except ValueError:
            # MNE-Python type file
            bads = lines
        else:
            # Maxfilter type file
            if len(lines) > 1:
                raise RuntimeError('Could not parse bad file')
            bads = lines[0].split()
            if len(bads) > 0:
                for fmt in ('MEG %03d', 'MEG%04d'):
                    if fmt % int(bads[0]) in raw.ch_names:
                        break
                bads = [fmt % int(bad) for bad in bads]
    else:
        bads = list()
    extra = 'additional ' if append else ''
    if disp:
        pl = '' if len(bads) == 1 else 's'
        print('%sMarking %s %sbad MEG channel%s using %s'
              % (prefix, len(bads), extra, pl, op.basename(prebad_file)))
    if append:
        bads = sorted(set(raw.info['bads']).union(bads))
    raw.info['bads'] = bads
    raw.info._check_consistency()


def _pos_valid(pos, dist_limit, gof_limit):
    """Check for a usable head position."""
    a = (np.abs(pos[..., 1:4]).max(axis=-1) <= 1)  # all abs(quats) <= 1
    b = (np.linalg.norm(pos[..., 4:7], axis=-1) <= 10)  # all pos < 10 m
    c = ((pos[..., 1:7] != 0).any(axis=-1))  # actual quat+pos
    d = (pos[..., 7] >= gof_limit)  # decent GOF
    e = (pos[..., 8] <= dist_limit)  # decent dists
    return a & b & c & d & e


def _get_t_window(p, raw):
    t_window = p.coil_t_window if p is not None else 'auto'
    if t_window == 'auto':
        from mne.chpi import _get_hpi_info
        try:
            hpi_freqs, _, _ = _get_hpi_info(raw.info, verbose=False)
        except RuntimeError:
            t_window = 0.2
        else:
            # Use the longer of 5 cycles and the difference in HPI freqs.
            # This will be 143 ms for 7 Hz spacing (old) and
            # 60 ms for 83 Hz lowest freq.
            t_window = max(5. / min(hpi_freqs), 1. / np.diff(hpi_freqs).min())
            t_window = round(1000 * t_window) / 1000.  # round to ms
    t_window = float(t_window)
    return t_window


def _maxbad(p, raw, bad_file):
    """Handle Maxfilter bad channels selection prior to SSS."""
    if op.isfile(bad_file):
        return
    print('        Generating MaxFilter bad file using %s: %s'
          % (p.mf_autobad_type, op.basename(bad_file),))
    skip_start = raw._first_time
    if raw.times[-1] > 45:
        skip_stop = skip_start + 30.
    elif raw.times[-1] > 10:
        skip_stop = skip_start + 1
    else:
        skip_stop = None
    func = dict(
        python=_python_autobad, maxfilter=_mf_autobad)[p.mf_autobad_type]
    bads = func(raw, p, skip_start, skip_stop)
    bads = ' '.join(['%04d' % (bad,) for bad in sorted(bads)])
    with open(bad_file, 'w') as f:
        f.write(bads)


def _python_autobad(raw, p, skip_start, skip_stop):
    from mne.preprocessing import mark_flat, find_bad_channels_maxwell
    raw = raw.copy()
    if skip_stop is not None:
        skip_stop = skip_stop - raw._first_time
        raw = raw.crop(skip_stop, None)
    with use_log_level(False):
        raw.load_data().fix_mag_coil_types()
    del skip_stop, skip_start
    orig_bads = raw.info['bads']
    picks_meg = pick_types(raw.info)
    mark_flat(raw, picks=picks_meg, verbose=False)
    flats = [bad for bad in raw.info['bads'] if bad not in orig_bads]
    if raw.info['dev_head_t'] is None:
        coord_frame, origin = 'meg', (0., 0., 0.)
    else:
        coord_frame, origin = 'head', _get_origin(p, raw)[0]
    filter_chpi(raw, t_window=_get_t_window(p, raw), allow_line_only=True,
                verbose=False)
    cal_file, ct_file = _get_cal_ct_file(p)
    bads, flats = find_bad_channels_maxwell(
        raw, p.mf_badlimit, origin=origin, coord_frame=coord_frame,
        bad_condition='warning', calibration=cal_file,
        cross_talk=ct_file, verbose=False)
    bads += flats
    assert all(len(bad) in (7, 8) for bad in bads)
    assert all(bad.startswith('MEG') for bad in bads)
    bads = sorted(int(bad.lstrip('MEG').lstrip(' ').lstrip('0'))
                  for bad in (bads + flats))
    return bads


def _mf_autobad(raw, p, skip_start, skip_stop):
    # Options used that matter for Python equivalence:
    # -skip, -bad, -frame, -origin
    opts = ('-v -force -badlimit %d -autobad on -format short'
            % (p.mf_badlimit,))
    if skip_stop is not None:
        opts += ' -skip %d %d' % (skip_start, skip_stop)
    if len(raw.info['bads']) > 0:
        bads = [bad.split('MEG')[1].strip()
                for bad in raw.info['bads'] if bad.startswith('MEG')]
        if len(bads) > 0:
            opts += ' -bad %s' % (' '.join(bads))
    # print(' (using %s)' % (opts,))
    kwargs = dict(
        host=p.sws_ssh, work_dir=p.sws_dir, port=p.sws_port,
        prefix=' ' * 10, verbose='error')
    if raw.info['dev_head_t'] is None:
        frame_opts = ' -frame device -origin 0 0 0'
    else:
        origin, _, _ = _get_origin(p, raw)
        frame_opts = (' -frame head -origin %0.1f %0.1f %0.1f'
                      % tuple(1000 * origin))
        del origin
    stdout, err, code = run_sss_command(
        raw, opts + frame_opts, None, throw_error=False, **kwargs)
    if code != 0:
        if 'outside of the helmet' in err and 'head' in frame_opts:
            warnings.warn('Head origin was outside the helmet, re-running '
                          'using device origin')
            frame_opts = ' -frame device -origin 0 0 0'
            stdout, err = run_sss_command(
                raw, opts + frame_opts, None, **kwargs)
        else:
            raise RuntimeError(
                'Maxbad failed (%d):\nSTDOUT:\n\n%s\n\nSTDERR:\n%s'
                % (code, stdout, err))
    output = stdout.splitlines()
    del stdout, err, code
    # Parse output for bad channels
    bads = set()
    for line in output:
        if 'Static bad channels' in line:
            bads = bads.union(line.split(':')[1].strip().split())
        if 'flat channels:' in line:
            bads = bads.union(line.split(':')[1].strip().split())
    bads = set(int(bad) for bad in bads)
    old_bads = [int(bad.lstrip('MEG').lstrip(' ').lstrip('0'))
                for bad in raw.info['bads'] if bad.startswith('MEG')]
    bads = bads.difference(set(old_bads))
    return bads


def _get_fit_data(raw, p=None, prefix='    '):
    if hasattr(p, 'tmpdir'):
        # Make these more tolerant and less frequent for faster runs
        count_fname = op.join(p.tmpdir, 'temp-counts.h5')
        locs_fname = op.join(p.tmpdir, 'temp-chpi_locs.h5')
        pos_fname = op.join(p.tmpdir, 'temp.pos')
    else:
        count_fname = raw.filenames[0][:-4] + '-counts.h5'
        locs_fname = raw.filenames[0][:-4] + '-chpi_locs.h5'
        pos_fname = raw.filenames[0][:-4] + '.pos'
    coil_dist_limit = p.coil_dist_limit
    coil_gof_limit = p.coil_gof_limit
    coil_t_step_min = p.coil_t_step_min
    if any(x is None for x in (p.movecomp, coil_dist_limit)):
        return None, None
    t_window = _get_t_window(p, raw)

    # Good to do the fits
    if not op.isfile(count_fname):
        fit_t, counts, n_coils, chpi_locs = compute_good_coils(
            raw, coil_t_step_min, t_window, coil_dist_limit,
            prefix=prefix, gof_limit=coil_gof_limit, verbose=True)
        write_hdf5(locs_fname, chpi_locs, title='mnefun')
        write_hdf5(count_fname,
                   dict(fit_t=fit_t, counts=counts, n_coils=n_coils,
                        t_step=coil_t_step_min, t_window=t_window,
                        coil_dist_limit=coil_dist_limit), title='mnefun')
    fit_data = read_hdf5(count_fname, 'mnefun')
    if op.isfile(locs_fname):
        chpi_locs = read_hdf5(locs_fname, 'mnefun')
    else:
        chpi_locs = None
    for key, val in (('t_step', coil_t_step_min),
                     ('t_window', t_window),
                     ('coil_dist_limit', coil_dist_limit)):
        if fit_data[key] != val:
            raise RuntimeError('Data mismatch %s (%s != %s), set '
                               'to match existing file or delete it:\n%s'
                               % (key, val, fit_data[key], count_fname))

    # head positions
    if not op.isfile(pos_fname):
        print('%sEstimating position file %s using %s'
              % (prefix, op.basename(pos_fname), p.hp_type))
        if p.hp_type == 'maxfilter':
            opts = ' '.join(bad.strip('MEG').strip()
                            for bad in raw.info['bads']
                            if bad.startswith('MEG'))
            if opts:
                opts = '-bad ' + opts + ' '
            opts += '-autobad off -force'
            run_sss_positions(raw.filenames[0], pos_fname,
                              host=p.sws_ssh, port=p.sws_port, prefix=prefix,
                              work_dir=p.sws_dir, t_window=t_window,
                              t_step_min=coil_t_step_min, opts=opts,
                              dist_limit=coil_dist_limit,
                              gof_limit=coil_gof_limit)
        else:
            assert p.hp_type == 'python'
            from mne.chpi import compute_head_pos
            if chpi_locs is None:
                raise RuntimeError('When using Python mode, delete existing '
                                   '-annot.fif files so that coil locations '
                                   'are recomputed')
            head_pos = compute_head_pos(
                raw.info, chpi_locs, coil_dist_limit, coil_gof_limit)
            write_head_pos(pos_fname, head_pos)
    head_pos = read_head_pos(pos_fname)

    # otherwise we need to go back and fix!
    assert _pos_valid(head_pos[0], coil_dist_limit, coil_gof_limit), pos_fname

    return fit_data, head_pos


def _head_pos_annot(p, subj, raw_fname, prefix='  '):
    """Locate head position estimation file and do annotations."""
    raw = _read_raw_prebad(p, subj, raw_fname, disp=False)
    from mne.annotations import _handle_meas_date
    printed = False
    if p is not None and p.movecomp is None:
        head_pos = fit_data = None
    else:
        t_window = _get_t_window(p, raw)
        # do the coil counts
        fit_data, head_pos = _get_fit_data(raw, p, prefix)

    # do the annotations
    annot_fname = raw_fname[:-4] + '-annot.fif'
    if not op.isfile(annot_fname) and fit_data is not None:
        lims = [p.rotation_limit, p.translation_limit, p.coil_dist_limit,
                p.coil_t_step_min, t_window, p.coil_bad_count_duration_limit]
        if np.isfinite(lims[:3]).any() or np.isfinite(lims[5]):
            print(prefix.join(['', 'Annotating raw segments with:\n',
                               u'  rotation_limit    = %s °/s\n' % lims[0],
                               u'  translation_limit = %s m/s\n' % lims[1],
                               u'  coil_dist_limit   = %s m\n' % lims[2],
                               u'  t_step, t_window  = %s, %s sec\n'
                               % (lims[3], lims[4]),
                               u'  3-good limit      = %s sec' % (lims[5],)]))
        annot = annotate_head_pos(
            raw, head_pos, rotation_limit=lims[0], translation_limit=lims[1],
            fit_t=fit_data['fit_t'], counts=fit_data['counts'],
            prefix='  ' + prefix, coil_bad_count_duration_limit=lims[5])
        if annot is not None:
            annot.save(annot_fname)
        printed = True
    orig_time = _handle_meas_date(raw.info['meas_date'])
    if op.isfile(annot_fname):
        annot = read_annotations(annot_fname)
        assert annot.orig_time is None  # relative to start of raw data
        annot.onset += raw.first_samp / raw.info['sfreq']
        try:
            annot.orig_time = orig_time
        except AttributeError:
            annot._orig_time = orig_time
    else:
        annot = Annotations([], [], [], orig_time=orig_time)

    # Append custom annotations (probably needs some tweaking due to meas_date)
    custom_fname = raw_fname[:-4] + '-custom-annot.fif'
    if op.isfile(custom_fname):
        custom_annot = read_annotations(custom_fname)
        assert custom_annot.orig_time == orig_time
        if printed:  # only do this if we're recomputing something
            print(prefix + 'Using custom annotations: %s'
                  % (op.basename(custom_fname),))
    else:
        custom_annot = Annotations([], [], [], orig_time=orig_time)
    assert custom_annot.orig_time == orig_time
    annot += custom_annot
    return head_pos, annot, fit_data


def info_sss_basis(info, origin='auto', int_order=8, ext_order=3,
                   coord_frame='head', regularize='in', ignore_ref=True):
    """Compute the SSS basis for a given measurement info structure

    Parameters
    ----------
    info : instance of io.Info
        The measurement info.
    origin : array-like, shape (3,) | str
        Origin of internal and external multipolar moment space in meters.
        The default is ``'auto'``, which means a head-digitization-based
        origin fit when ``coord_frame='head'``, and ``(0., 0., 0.)`` when
        ``coord_frame='meg'``.
    int_order : int
        Order of internal component of spherical expansion.
    ext_order : int
        Order of external component of spherical expansion.
    coord_frame : str
        The coordinate frame that the ``origin`` is specified in, either
        ``'meg'`` or ``'head'``. For empty-room recordings that do not have
        a head<->meg transform ``info['dev_head_t']``, the MEG coordinate
        frame should be used.
    destination : str | array-like, shape (3,) | None
        The destination location for the head. Can be ``None``, which
        will not change the head position, or a string path to a FIF file
        containing a MEG device<->head transformation, or a 3-element array
        giving the coordinates to translate to (with no rotations).
        For example, ``destination=(0, 0, 0.04)`` would translate the bases
        as ``--trans default`` would in MaxFilter™ (i.e., to the default
        head location).
    regularize : str | None
        Basis regularization type, must be "in", "svd" or None.
        "in" is the same algorithm as the "-regularize in" option in
        MaxFilter™. "svd" (new in v0.13) uses SVD-based regularization by
        cutting off singular values of the basis matrix below the minimum
        detectability threshold of an ideal head position (usually near
        the device origin).
    ignore_ref : bool
        If True, do not include reference channels in compensation. This
        option should be True for KIT files, since Maxwell filtering
        with reference channels is not currently supported.
    """
    from mne.io import pick_info
    from mne.preprocessing.maxwell import \
        _check_origin, _check_regularize, _get_mf_picks, _prep_mf_coils, \
        _trans_sss_basis, _prep_regularize, _regularize
    if coord_frame not in ('head', 'meg'):
        raise ValueError('coord_frame must be either "head" or "meg", not "%s"'
                         % coord_frame)
    origin = _check_origin(origin, info, 'head')
    regularize = _check_regularize(regularize, ('in', 'svd'))
    meg_picks, mag_picks, grad_picks, good_picks, coil_scale, mag_or_fine = \
        _get_mf_picks(info, int_order, ext_order, ignore_ref)
    info_good = pick_info(info, good_picks, copy=True)
    all_coils = _prep_mf_coils(info_good, ignore_ref=ignore_ref)
    # remove MEG bads in "to" info
    decomp_coil_scale = coil_scale[good_picks]
    exp = dict(int_order=int_order, ext_order=ext_order, head_frame=True,
               origin=origin)
    # prepare regularization techniques
    if _prep_regularize is None:
        raise RuntimeError('mne-python needs to be on the experimental SVD '
                           'branch to use this function')
    _prep_regularize(regularize, all_coils, None, exp, ignore_ref,
                     coil_scale, grad_picks, mag_picks, mag_or_fine)
    # noinspection PyPep8Naming
    S = _trans_sss_basis(exp, all_coils, info['dev_head_t'],
                         coil_scale=decomp_coil_scale)
    if regularize is not None:
        # noinspection PyPep8Naming
        S = _regularize(regularize, exp, S, mag_or_fine, t=0.)[0]
    S /= np.linalg.norm(S, axis=0)
    return S


def calc_median_hp(p, subj, out_file, ridx):
    """Calculate median head position"""
    print('        Estimating median head position ...')
    from mne.io.meas_info import _empty_info
    raw_files = get_raw_fnames(p, subj, 'raw', False, False, ridx)
    ts = []
    qs = []
    info = None
    for fname in raw_files:
        info = read_info(fname)
        trans = info['dev_head_t']['trans']
        ts.append(trans[:3, 3])
        m = trans[:3, :3]
        # make sure we are a rotation matrix
        assert_allclose(np.dot(m, m.T), np.eye(3), atol=1e-5)
        assert_allclose(np.linalg.det(m), 1., atol=1e-5)
        qs.append(rot_to_quat(m))
    assert info is not None
    if len(raw_files) == 1:  # only one head position
        dev_head_t = info['dev_head_t']
    else:
        t = np.median(np.array(ts), axis=0)
        rot = np.median(quat_to_rot(np.array(qs)), axis=0)
        trans = np.r_[np.c_[rot, t[:, np.newaxis]],
                      np.array([0, 0, 0, 1], t.dtype)[np.newaxis, :]]
        dev_head_t = {'to': 4, 'from': 1, 'trans': trans}
    info = _empty_info(info['sfreq'])
    info['dev_head_t'] = dev_head_t
    write_info(out_file, info)


def calc_twa_hp(p, subj, out_file, ridx):
    """Calculate time-weighted average head position."""
    from mne.annotations import _annotations_starts_stops
    from mne.io.meas_info import _empty_info
    if not p.movecomp:
        # Eventually we could relax this but probably YAGNI
        raise RuntimeError('Cannot use time-weighted average head position '
                           'when movecomp is off.')
    print('        Estimating time-weighted average head position ...')
    raw_fnames = get_raw_fnames(p, subj, 'raw', False, False, ridx)
    assert len(raw_fnames) >= 1
    norm = 0
    A = np.zeros((4, 4))
    pos = np.zeros(3)
    for raw_fname in raw_fnames:
        raw = read_raw_fif(raw_fname, allow_maxshield='yes', verbose='error')
        hp, annot, _ = _head_pos_annot(p, subj, raw_fname, prefix='          ')
        try:
            raw.set_annotations(annot)
        except AttributeError:
            raw.annotations = annot
        good = np.ones(len(raw.times))
        hp_ts = hp[:, 0] - raw.first_samp / raw.info['sfreq']
        if hp_ts[0] < 0:
            hp_ts[0] = 0
            assert hp_ts[1] > 1. / raw.info['sfreq']
        mask = hp_ts <= raw.times[-1]
        if not mask.all():
            warnings.warn(
                '          Removing %0.1f%% time points > raw.times[-1] (%s)'
                % ((~mask).sum() / float(len(mask)), raw.times[-1]))
            hp = hp[mask]
        del mask, hp_ts
        ts = np.concatenate((hp[:, 0],
                             [(raw.last_samp + 1) / raw.info['sfreq']]))
        assert (np.diff(ts) > 0).all()
        ts -= raw.first_samp / raw.info['sfreq']
        idx = raw.time_as_index(ts, use_rounding=True)
        del ts
        if idx[0] == -1:  # annoying rounding errors
            idx[0] = 0
            assert idx[1] > 0
        assert (idx >= 0).all()
        assert idx[-1] == len(good)
        assert (np.diff(idx) > 0).all()
        # Mark times bad that are bad according to annotations
        onsets, ends = _annotations_starts_stops(raw, 'bad')
        for onset, end in zip(onsets, ends):
            good[onset:end] = 0
        dt = np.diff(np.cumsum(np.concatenate([[0], good]))[idx])
        assert (dt >= 0).all()
        dt = dt / raw.info['sfreq']
        del good, idx
        pos += np.dot(dt, hp[:, 4:7])
        these_qs = hp[:, 1:4]
        res = 1 - np.sum(these_qs * these_qs, axis=-1, keepdims=True)
        assert (res >= 0).all()
        these_qs = np.concatenate((these_qs, np.sqrt(res)), axis=-1)
        assert np.allclose(np.linalg.norm(these_qs, axis=1), 1)
        these_qs *= dt[:, np.newaxis]
        # rank 1 update method
        # https://arc.aiaa.org/doi/abs/10.2514/1.28949?journalCode=jgcd
        # https://github.com/tolgabirdal/averaging_quaternions/blob/master/wavg_quaternion_markley.m  # noqa: E501
        # qs.append(these_qs)
        outers = np.einsum('ij,ik->ijk', these_qs, these_qs)
        A += outers.sum(axis=0)
        dt_sum = dt.sum()
        assert dt_sum >= 0
        norm += dt_sum
    if norm <= 0:
        raise RuntimeError('No good segments found (norm=%s)' % (norm,))
    A /= norm
    best_q = linalg.eigh(A)[1][:, -1]  # largest eigenvector is the wavg
    # Same as the largest eigenvector from the concatenation of all
    # best_q = linalg.svd(np.concatenate(qs).T)[0][:, 0]
    best_q = best_q[:3] * np.sign(best_q[-1])
    trans = np.eye(4)
    trans[:3, :3] = quat_to_rot(best_q)
    trans[:3, 3] = pos / norm
    assert np.linalg.norm(trans[:3, 3]) < 1  # less than 1 meter is sane
    dev_head_t = Transform('meg', 'head', trans)
    info = _empty_info(raw.info['sfreq'])
    info['dev_head_t'] = dev_head_t
    write_info(out_file, info)


def _old_chpi_locs(raw, t_step, t_window, prefix):
    # XXX we can remove this once people are on 0.20+
    from mne.chpi import (_get_hpi_initial_fit, _setup_hpi_struct,
                          _fit_cHPI_amplitudes, _fit_magnetic_dipole)
    hpi_dig_head_rrs = _get_hpi_initial_fit(raw.info, verbose=False)
    n_window = (int(round(t_window * raw.info['sfreq'])) // 2) * 2 + 1
    del t_window
    hpi = _setup_hpi_struct(raw.info, n_window, verbose=False)
    n_step = int(round(t_step * raw.info['sfreq']))
    del t_step
    starts = np.arange(0, len(raw.times) - n_window // 2, n_step)
    head_dev_t = invert_transform(
        raw.info['dev_head_t'])['trans']
    coil_dev_rrs = apply_trans(head_dev_t, hpi_dig_head_rrs)
    last_fit = None
    last = -10.
    logger.info('%sComputing %d coil fits in %0.1f ms steps over %0.1f sec'
                % (prefix, len(starts), (n_step / raw.info['sfreq']) * 1000,
                   raw.times[-1]))
    times, rrs, gofs = list(), list(), list()
    for ii, start in enumerate(starts):
        time_sl = slice(max(start - n_window // 2, 0), start + n_window // 2)
        t = start / raw.info['sfreq']
        if t - last >= 10. - 1e-7:
            logger.info('%s    Fitting %0.1f - %0.1f sec'
                        % (prefix, t, min(t + 10., raw.times[-1])))
            last = t
        # Ignore warnings about segments with not enough coils on
        sin_fit = _fit_cHPI_amplitudes(raw, time_sl, hpi, t, verbose=False)
        # skip this window if it bad.
        if sin_fit is None:
            continue

        # check if data has sufficiently changed
        if last_fit is not None:  # first iteration
            # The sign of our fits is arbitrary
            flips = np.sign((sin_fit * last_fit).sum(-1, keepdims=True))
            sin_fit *= flips
            corr = np.corrcoef(sin_fit.ravel(), last_fit.ravel())[0, 1]
            # check to see if we need to continue
            if corr * corr > 0.98:
                # don't need to refit data
                continue

        last_fit = sin_fit.copy()

        kwargs = dict()
        if 'too_close' in get_args(_fit_magnetic_dipole):
            kwargs['too_close'] = 'warning'

        outs = [_fit_magnetic_dipole(f, pos, hpi['coils'], hpi['scale'],
                                     hpi['method'], **kwargs)
                for f, pos in zip(sin_fit, coil_dev_rrs)]

        rr, gof = zip(*outs)
        rrs.append(rr)
        gofs.append(gof)
        times.append(t)
    return dict(rrs=np.array(rrs, float), gofs=np.array(gofs, float),
                times=np.array(times, float))


@verbose
def compute_good_coils(raw, t_step=0.01, t_window=0.2, dist_limit=0.005,
                       prefix='', gof_limit=0.98, verbose=None):
    """Comute time-varying coil distances."""
    try:
        from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs
    except ImportError:
        chpi_locs = _old_chpi_locs(raw, t_step, t_window, prefix)
    else:
        chpi_amps = compute_chpi_amplitudes(
            raw, t_step_min=t_step, t_window=t_window)
        chpi_locs = compute_chpi_locs(raw.info, chpi_amps)
    from mne.chpi import _get_hpi_initial_fit
    hpi_dig_head_rrs = _get_hpi_initial_fit(raw.info, verbose=False)
    hpi_coil_dists = cdist(hpi_dig_head_rrs, hpi_dig_head_rrs)

    counts = np.empty(len(chpi_locs['times']), int)
    for ii, (t, coil_dev_rrs, gof) in enumerate(zip(
            chpi_locs['times'], chpi_locs['rrs'], chpi_locs['gofs'])):
        these_dists = cdist(coil_dev_rrs, coil_dev_rrs)
        these_dists = np.abs(hpi_coil_dists - these_dists)
        # there is probably a better algorithm for finding the bad ones...
        use_mask = gof >= gof_limit
        good = False
        while not good:
            d = these_dists[use_mask][:, use_mask]
            d_bad = d > dist_limit
            good = not d_bad.any()
            if not good:
                if use_mask.sum() == 2:
                    use_mask[:] = False
                    break  # failure
                # exclude next worst point
                badness = (d * d_bad).sum(axis=0)
                exclude_coils = np.where(use_mask)[0][np.argmax(badness)]
                use_mask[exclude_coils] = False
        counts[ii] = use_mask.sum()
    t = chpi_locs['times'] - raw.first_samp / raw.info['sfreq']
    return t, counts, len(hpi_dig_head_rrs), chpi_locs


def annotate_head_pos(raw, head_pos, rotation_limit=45, translation_limit=0.1,
                      fit_t=None, counts=None, prefix='  ',
                      coil_bad_count_duration_limit=0.1):
    u"""Annotate a raw instance based on coil counts and head positions.

    Parameters
    ----------
    raw : instance of Raw
        The raw instance.
    head_pos : ndarray | None
        The head positions. Can be None if movement compensation is off
        to short-circuit the function.
    rotation_limit : float
        The rotational velocity limit in °/s.
        Can be infinite to skip rotation checks.
    translation_limit : float
        The translational velocity limit in m/s.
        Can be infinite to skip translation checks.
    fit_t : ndarray
        Fit times.
    counts : ndarray
        Coil counts.
    prefix : str
        The prefix for printing.
    coil_bad_count_duration_limit : float
        The lower limit for bad coil counts to remove segments of data.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """
    from mne.utils import _mask_to_onsets_offsets
    # XXX: Add `sphere_dist_limit` to ensure no sensor collisions at some
    # point
    do_rotation = np.isfinite(rotation_limit) and head_pos is not None
    do_translation = np.isfinite(translation_limit) and head_pos is not None
    do_coils = (fit_t is not None and
                counts is not None and
                np.isfinite(coil_bad_count_duration_limit))
    if not (do_rotation or do_translation or do_coils):
        return None
    head_pos_t = head_pos[:, 0].copy()
    head_pos_t -= raw.first_samp / raw.info['sfreq']
    dt = np.diff(head_pos_t)
    head_pos_t = np.concatenate([head_pos_t,
                                 [head_pos_t[-1] + 1. / raw.info['sfreq']]])

    annot = Annotations([], [], [], orig_time=None)  # rel to data start

    # Annotate based on bad coil distances
    if do_coils:
        fit_t = np.concatenate([fit_t, fit_t[-1] + [1. / raw.info['sfreq']]])
        bad_mask = (counts < 3)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets = fit_t[onsets]
        offsets = fit_t[offsets]
        count = 0
        dur = 0.
        for onset, offset in zip(onsets, offsets):
            if offset - onset > coil_bad_count_duration_limit - 1e-6:
                annot.append(onset, offset - onset, 'BAD_HPI_COUNT')
                dur += offset - onset
                count += 1
        print('%sOmitting %5.1f%% (%3d segments): '
              '< 3 good coils for over %s sec'
              % (prefix, 100 * dur / raw.times[-1], count,
                 coil_bad_count_duration_limit))

    # Annotate based on rotational velocity
    t_tot = raw.times[-1]
    if do_rotation:
        from mne.transforms import _angle_between_quats
        assert rotation_limit > 0
        # Rotational velocity (radians / sec)
        r = _angle_between_quats(head_pos[:-1, 1:4], head_pos[1:, 1:4])
        r /= dt
        bad_mask = (r >= np.deg2rad(rotation_limit))
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = head_pos_t[onsets], head_pos_t[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        print(u'%sOmitting %5.1f%% (%3d segments): '
              u'ω >= %5.1f°/s (max: %0.1f°/s)'
              % (prefix, bad_pct, len(onsets), rotation_limit,
                 np.rad2deg(r.max())))
        for onset, offset in zip(onsets, offsets):
            annot.append(onset, offset - onset, 'BAD_RV')

    # Annotate based on translational velocity
    if do_translation:
        assert translation_limit > 0
        v = np.linalg.norm(np.diff(head_pos[:, 4:7], axis=0), axis=-1)
        v /= dt
        bad_mask = (v >= translation_limit)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = head_pos_t[onsets], head_pos_t[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        print(u'%sOmitting %5.1f%% (%3d segments): '
              u'v >= %5.4fm/s (max: %5.4fm/s)'
              % (prefix, bad_pct, len(onsets), translation_limit, v.max()))
        for onset, offset in zip(onsets, offsets):
            annot.append(onset, offset - onset, 'BAD_TV')

    # Annotate on distance from the sensors
    return annot


def check_sws():
    """Check if SSS workstation is configured correctly."""
    from . import Params
    p = Params()  # loads config
    if p.sws_ssh is None:
        raise RuntimeError('sws_ssh was not defined in mnefun config file %s'
                           % (_get_config_file(),))
    output = run_sss_command(None, '-version', None, host=p.sws_ssh,
                             work_dir=p.sws_dir, port=p.sws_port,
                             verbose=False)
    output = [o.strip() for o in output]
    if output[0]:
        print('Output:\n%s' % (output[0].strip('$').strip(),))
    if output[1]:
        print('ERRORS:\n%s' % (output[1],))
