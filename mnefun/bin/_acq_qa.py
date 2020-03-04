#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitor acquisition paths for new files and generate reports for them.
"""

import argparse
from datetime import datetime
import logging
import os
import os.path as op
import time
import traceback

import numpy as np

import mne

logger = logging.getLogger('mnefun.acq_qa')


def acq_qa():
    """Run acquisition QA."""
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+', help='Path(s) to monitor')
    parser.add_argument('--write-root', '-r', dest='write_root', default='/',
                        help='Root directory for writing the reports')
    parser.add_argument('--exclude', '-x', dest='exclude', default='',
                        help='Comma-separated list of patterns to exclude')
    parser.add_argument('--quit', '-q', dest='quit_on_error',
                        action="store_true", help="Quit on error",
                        default=False)
    args = parser.parse_args()
    write_root = op.abspath(op.expanduser(args.write_root))
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]

    # Logging
    logger.setLevel(logging.DEBUG)
    log_dir = op.expanduser('~/log')
    os.makedirs(log_dir, exist_ok=True)
    log_fname = op.join(
        log_dir, 'acq_qa_%s.log' % (datetime.now().isoformat(),))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_fname)
    fh.setLevel(logging.INFO)
    exclude = args.exclude.split(',')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    for h in (ch, fh):
        h.setFormatter(formatter)
        logger.addHandler(h)
    if len(full_paths) == 0:
        print('No paths provided, exiting')
        return 1
    while True:  # this thing runs forever...
        try:
            for path in full_paths:
                _walk_path(path, write_root, args.quit_on_error, exclude)
            logger.debug('Sleeping for 10 seconds...')
            time.sleep(10.)
        except KeyboardInterrupt:
            logger.info('%s\n\nCaught keyboard interrupt above, exiting '
                        'normally', traceback.format_exc())
            return 0
        except:  # noqa
            logger.error('%s\n\nCaught error above, exiting with status 1',
                         traceback.format_exc())
            return 1


def _check_exclude(path, exclude):
    for e in exclude:
        if e in path:
            logger.debug('  Skipping %s (exclude %r)' % (path, e))
            return True
    return False


def _walk_path(path, write_root, quit_on_error, exclude):
    logger.debug('Traversing %s' % (path,))
    # The [::-1] here helps ensure that we go in reverse chronological
    # order for empty-room recordings (most recent first)
    for root, dirs, files in sorted(os.walk(path, topdown=True))[::-1]:
        if _check_exclude(root, exclude):
            continue
        logger.debug('  %s', root)
        for fname in sorted(files):
            if _check_exclude(fname, exclude):
                continue
            # skip if wrong ext
            if not fname.endswith('_raw.fif'):
                continue
            # skip if report done
            report_fname = op.join(
                write_root + root, op.splitext(fname)[0] + '.html')
            if op.isfile(report_fname):
                continue
            # skip if it has been deemed unreadable previously
            skip_report_fname = op.join(
                write_root + root, '.' + op.splitext(fname)[0] + '.html')
            if op.isfile(skip_report_fname):
                continue
            raw_fname = op.join(root, fname)
            try:
                os.stat(raw_fname)
            except FileNotFoundError:
                # Can happen for links, which can't necessarily be detected
                # by op.islink
                continue
            # skip if modified time is within the last 10 seconds
            mtime = os.path.getmtime(raw_fname)
            delta = time.time() - mtime
            if delta < 10:
                logger.info('    Skipping file modified %0.1f sec ago: %s',
                            delta, fname)
            # skip if not a raw instance
            try:
                raw = mne.io.read_raw_fif(
                    raw_fname, allow_maxshield='yes', verbose=False)
            except Exception:
                err = traceback.format_exc()
                logger.debug(
                    '    Skipping file that cannot be read %s:\n%s',
                    fname, err)
                os.makedirs(op.dirname(skip_report_fname), exist_ok=True)
                with open(skip_report_fname, 'w') as fid:
                    fid.write(err)
                continue
            del raw_fname
            # actually generate the report!
            logger.info('Generating %s' % (report_fname,))
            _flush_log()
            _generate_report(raw, report_fname, quit_on_error)
            logger.info('Done with %s' % (report_fname,))
            _flush_log()


def _flush_log():
    for lh in logger.handlers:
        lh.flush()


_HTML_TEMPLATE = """
<div style="text-align:center;"><h5>{title}</h5><p>{text}</p></div>
"""


def _generate_report(raw, report_fname, quit_on_error):
    from .._mnefun import _set_static
    from .._sss import _maxbad, _load_meg_bads
    from .._report import (report_context, _report_good_hpi, _report_chpi_snr,
                           _report_head_movement, _report_raw_segments,
                           _report_events, _report_raw_psd)
    report = mne.Report(verbose=False)
    raw.load_data()
    with report_context():
        import matplotlib.pyplot as plt
        p = mne.utils.Bunch(
            mf_badlimit=7, tmpdir=mne.utils._TempDir(),
            coil_dist_limit=0.01, coil_t_window='auto', coil_gof_limit=0.95,
            coil_t_step_min=0.01, lp_trans=10, lp_cut=40, movecomp=True,
            hp_type='python', coil_bad_count_duration_limit=np.inf)
        maxbad_file = op.join(p.tmpdir, 'maxbad.txt')
        _set_static(p)
        _maxbad(p, raw, maxbad_file)
        # Maxbads
        _load_meg_bads(raw, maxbad_file, disp=False)
        section = 'MF Autobad'
        htmls = _HTML_TEMPLATE.format(
            title='%d bad channel%s detected' % (
                len(raw.info['bads']), mne.utils._pl(raw.info['bads'])),
            text=', '.join(raw.info['bads'],))
        report.add_htmls_to_section(htmls, section, section)
        # HPI count, SNR, head position
        funcs = (
            [_report_good_hpi, 'Good HPI count'],
            [_report_chpi_snr, 'cHPI SNR'],
            [_report_head_movement, 'Head movement'],
            [_report_events, 'Events'],
        )
        if raw.info['dev_head_t'] is None:  # don't even try the first three
            funcs = funcs[3:]
        for func, section in funcs:
            try:
                func(report, [raw], p=p)
            except Exception as exp:
                if quit_on_error:
                    raise
                htmls = _HTML_TEMPLATE.format(title='Error', text=str(exp))
                report.add_htmls_to_section(htmls, section, section)
        # Raw segments (ignoring warnings about dev_head_t)
        with mne.utils.use_log_level('error'):
            _report_raw_segments(report, raw, lowpass=p.lp_cut)
        # Raw PSD
        _report_raw_psd(report, raw, p=p)
        os.makedirs(op.dirname(report_fname), exist_ok=True)
        report.save(report_fname, open_browser=False)
        plt.close('all')
