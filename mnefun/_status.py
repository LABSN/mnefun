# -*- coding: utf-8 -*-
# Copyright (c) 2015, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from os import path as op

from .externals import tabulate
from ._paths import (get_raw_fnames, get_event_fnames, get_cov_fwd_inv_fnames,
                     get_epochs_evokeds_fnames, get_report_fnames)
from ._reorder import _all_files_fixed


def _have_all(fnames):
    """Check to make sure all files exist"""
    return all(op.isfile(fname) for fname in fnames)


def print_proc_status(p, subjects, structurals, analyses):
    """Print status update"""
    steps_all = []
    status_mapping = dict(missing=' ',
                          complete='X',
                          unknown='?')

    # XXX TODO in subsequent PR:
    # * Add modified-date checks to make sure provenance is correct

    for subj, struc in zip(subjects, structurals):
        fetch_raw = do_score = prebads = coreg = fetch_sss = do_ch_fix = \
            do_ssp = apply_ssp = write_epochs = \
            gen_covs = gen_fwd = gen_inv = gen_report = 'missing'

        # check if raws fetched (+1 is for erm)
        if _have_all(get_raw_fnames(p, subj, 'raw')):
            fetch_raw = 'complete'

        # check if scoring has been done
        if _have_all(get_event_fnames(p, subj)):
            do_score = 'complete'

        # check if prebads created
        if op.isfile(op.join(subj, 'raw_fif', subj + '_prebad.txt')):
            prebads = 'complete'

        # check if coreg has been done
        if struc is None or \
                op.isfile(op.join(subj, 'trans', subj + '-trans.fif')):
            coreg = 'complete'

        # check if sss has been fetched (+1 is for erm)
        if _have_all(get_raw_fnames(p, subj, 'sss')):
            fetch_sss = 'complete'

        # check if channel orders have been fixed_all_files_fixed
        if _all_files_fixed(p, subj, 'sss'):
            do_ch_fix = 'complete'

        # check if SSPs have been generated
        if op.isfile(op.join(p.work_dir, subj, p.pca_dir,
                             'preproc_all-proj.fif')):
            do_ssp = 'complete'

        # check if SSPs have been applied:
        if _have_all(get_raw_fnames(p, subj, 'pca')):
            apply_ssp = 'complete'

        # check if epochs have been made
        epoch_fnames, evoked_fnames = get_epochs_evokeds_fnames(
            p, subj, analyses, remove_unsaved=True)
        if _have_all(epoch_fnames + evoked_fnames):
            write_epochs = 'complete'

        # check if covariance has been calculated
        cov_fnames, fwd_fnames, inv_fnames = get_cov_fwd_inv_fnames(p, subj)
        if _have_all(cov_fnames):
            gen_covs = 'complete'

        # check if forward solution has been calculated
        if _have_all(fwd_fnames):
            gen_fwd = 'complete'

        # check if inverses have been calculated
        if _have_all(inv_fnames):
            gen_inv = 'complete'

        # check if report has been made
        if _have_all(get_report_fnames(p, subj)):
            gen_report = 'complete'

        # Add up results
        these_steps = [
            fetch_raw, do_score, prebads, coreg,
            fetch_sss,
            do_ch_fix, do_ssp, apply_ssp, write_epochs,
            gen_covs, gen_fwd, gen_inv,
            gen_report,
        ]
        if all(s == 'complete' for s in these_steps):
            these_steps.append('complete')
        else:
            these_steps.append('missing')
        steps_all.append(these_steps)

    steps = ['raw', 'sco', 'pbd', 'coreg',
             'sss',
             'chfx', 'genssp', 'appssp', 'epevo',
             'cov', 'fwd', 'inv',
             'rep', 'done']
    assert all(len(steps) == len(s) for s in steps_all)

    # Print it out in a tabular manner
    headers = [''] + steps
    table = [[subj] + [status_mapping[key] for key in subj_steps]
             for subj, subj_steps in zip(subjects, steps_all)]
    print(tabulate.tabulate(table, headers, tablefmt='fancy_grid',
                            stralign='right'))
