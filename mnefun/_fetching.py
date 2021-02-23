"""Fetching raw files."""

import os
import os.path as op
import shutil
import subprocess
import warnings

from mne.utils import run_subprocess

from ._paths import get_raw_fnames, _regex_convert, _is_dir


def fetch_raw_files(p, subjects, run_indices):
    """Fetch remote raw recording files (only designed for *nix platforms)"""
    for si, subj in enumerate(subjects):
        print('  Checking for proper remote filenames for %s...' % subj)
        subj_dir = op.join(p.work_dir, subj)
        if not _is_dir(subj_dir):
            os.mkdir(subj_dir)
        raw_dir = op.join(subj_dir, p.raw_dir)
        if not op.isdir(raw_dir):
            os.mkdir(raw_dir)
        fnames = get_raw_fnames(p, subj, 'raw', True, False,
                                run_indices[si])
        assert len(fnames) > 0
        # build remote raw file finder
        if isinstance(p.acq_dir, str):
            use_dir = [p.acq_dir]
        else:
            use_dir = p.acq_dir
        finder_stem = 'find ' + ' '.join(use_dir)
        finder = (finder_stem + ' -o '.join([' -type f -regex ' +
                                             _regex_convert(f)
                                             for f in fnames]))
        # Ignore "Permission denied" errors:
        # https://unix.stackexchange.com/questions/42841/how-to-skip-permission-denied-errors-when-running-find-in-linux  # noqa
        finder += '2>&1 | grep -v "Permission denied"'
        stdout_ = run_subprocess(
            ['ssh', '-p', str(p.acq_port), p.acq_ssh, finder],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)[0]
        remote_fnames = [x.strip() for x in stdout_.splitlines()]
        if not any(fname.startswith(rd.rstrip('/') + '/') for rd in use_dir
                   for fname in remote_fnames):
            raise IOError('Unable to find files at remote locations. '
                          'Check filenames, for example:\n%s'
                          % remote_fnames[:1])
        # make the name "local" to the acq dir, so that the name works
        # remotely during rsync and locally during copyfile
        remote_dirs = sorted(set(
            fn[:fn.index(op.basename(fn))] for fn in remote_fnames))
        # sometimes there is more than one, for example if someone has done
        # some processing in the acquistion dir
        orig_remotes = remote_dirs
        remote_dirs = [remote_dir for remote_dir in remote_dirs
                       if not any(x in remote_dir for x in p.acq_exclude)]
        if len(remote_dirs) != 1:
            raise IOError('Unable to determine correct remote directory, got '
                          f'candidates:\n{remote_dirs}\n'
                          f'pruned from\n{orig_remotes}')
        remote_dir = remote_dirs[0]
        del orig_remotes, remote_dirs
        remote_fnames = sorted(set(
            op.basename(fname) for fname in remote_fnames))
        want = set(op.basename(fname) for fname in fnames)
        got = set(op.basename(fname) for fname in remote_fnames)
        if want != got.intersection(want):
            raise RuntimeError('Could not find all files, missing:\n' +
                               '\n'.join(sorted(want - got)))
        if len(remote_fnames) != len(fnames):
            warnings.warn('Found more files than expected on remote server.\n'
                          'Likely split files were found. Please confirm '
                          'results.')
        print('  Pulling %s files for %s...' % (len(remote_fnames), subj))
        cmd = ['rsync', '-avOe', 'ssh -p %s' % p.acq_port,
               '--no-perms', '--prune-empty-dirs', '--partial',
               '--include', '*/']
        for fname in remote_fnames:
            cmd += ['--include', op.basename(fname)]
        remote_loc = '%s:%s' % (p.acq_ssh, op.join(remote_dir, ''))
        cmd += ['--exclude', '*', remote_loc, op.join(raw_dir, '')]
        run_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # move files to root raw_dir
        for fname in remote_fnames:
            from_ = fname[fname.index(subj):].lstrip('/')
            to_ = op.basename(fname)
            if from_ != to_:  # can happen if it's at the root
                shutil.move(op.join(raw_dir, from_),
                            op.join(raw_dir, to_))
        # prune the extra directories we made
        for fname in remote_fnames:
            from_ = fname.index(subj)
            next_ = op.split(fname[from_:].lstrip('/'))[0]
            while len(next_) > 0:
                if op.isdir(op.join(raw_dir, next_)):
                    os.rmdir(op.join(raw_dir, next_))  # safe; goes if empty
                next_ = op.split(next_)[0]
