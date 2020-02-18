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

import mne

logger = logging.getLogger('mnefun.acq_qa')


def acq_qa():
    """Run acquisition QA."""
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+', help='Path(s) to monitor')
    parser.add_argument('--write-root', '-r', dest='write_root', default='/',
                        help='Root directory for writing the reports')
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
                _walk_path(path, write_root)
            logger.debug('Sleeping for 10 seconds...')
            time.sleep(10.)
        except KeyboardInterrupt:
            logger.info('Caught keyboard interrupt, exiting normally')
            return 0
        except:  # noqa
            logger.error('Caught error, exiting with status 1:\n%s',
                         traceback.format_exc())
            return 1


def _walk_path(path, write_root):
    logger.debug('Traversing %s' % (path,))
    for root, dirs, files in os.walk(path, topdown=True):
        logger.debug('  %s', root)
        for fname in files:
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
            # skip if modified time is within the last 10 seconds
            raw_fname = op.join(root, fname)
            mtime = os.path.getmtime(raw_fname)
            delta = time.time() - mtime
            if delta < 10:
                logger.info('Skipping file modified %0.1f sec ago: %s',
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
                with open(skip_report_fname, 'w') as fid:
                    fid.write(err)
                continue
            del raw_fname
            # actually generate the report!
            logger.info('Generating %s' % (report_fname,))
            _generate_report(raw, report_fname)
            logger.info('Done with %s' % (report_fname,))


def _generate_report(raw, report_fname):
    # XXX eventually we should maybe add head position estimation
    # and also add maxbads to the general one maybe
    from .._mnefun import _set_static
    from .._sss import _maxbad, _load_meg_bads
    from .._report import (report_context, _report_good_hpi, _report_chpi_snr,
                           _report_raw_segments, _report_raw_psd)
    report = mne.Report(verbose=False)
    raw.load_data()
    with report_context():
        tmpdir = mne.utils._TempDir()
        maxbad_file = op.join(tmpdir, 'maxbad.txt')
        p = mne.utils.Bunch(mf_badlimit=7)
        _set_static(p)
        _maxbad(p, raw, maxbad_file)
        _load_meg_bads(raw, maxbad_file, disp=False)
        section = 'MF Autobad'
        htmls = (
            '<div style="text-align:center;">'
            '<h5>%d bad channel%s detected</h5><p>%s</p></div>'
            % (len(raw.info['bads']), mne.utils._pl(raw.info['bads']),
               ', '.join(raw.info['bads'],)))
        report.add_htmls_to_section(htmls, section, section)
        _report_good_hpi(report, [raw])
        _report_chpi_snr(report, [raw])
        _report_raw_segments(report, raw, lowpass=40)
        _report_raw_psd(report, raw)
    os.makedirs(op.dirname(report_fname), exist_ok=True)
    report.save(report_fname, open_browser=False)
