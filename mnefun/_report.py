#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Create HTML reports."""
from __future__ import print_function, unicode_literals

from contextlib import contextmanager
from copy import deepcopy
import os.path as op
import time
import warnings

import numpy as np

import mne
from mne import read_proj, read_epochs
from mne.viz import plot_projs_topomap, plot_cov, plot_snr_estimate
from mne.viz._3d import plot_head_positions
from mne.report import Report
from mne.utils import _pl

from ._forward import _get_bem_src_trans
from ._paths import (get_raw_fnames, get_proj_fnames, get_report_fnames,
                     get_bad_fname, get_epochs_evokeds_fnames, safe_inserter)
from ._sss import (_load_trans_to, _head_pos_annot, _read_raw_prebad,
                   _get_t_window, _get_fit_data)
from ._viz import plot_good_coils, plot_chpi_snr_raw, trim_bg, mlab_offscreen
from ._utils import _fix_raw_eog_cals, _handle_dict

LJUST = 25


@contextmanager
def report_context():
    """Create a context for making plt and mlab figures."""
    import matplotlib
    import matplotlib.pyplot as plt
    style = {'axes.spines.right': 'off', 'axes.spines.top': 'off',
             'axes.grid': True}
    is_interactive = matplotlib.is_interactive()
    plt.ioff()
    old_backend = matplotlib.get_backend()
    matplotlib.use('Agg', force=True)
    try:
        with plt.style.context(style):
            yield
    except Exception:
        matplotlib.use(old_backend, force=True)
        plt.interactive(is_interactive)
        raise


def _report_good_hpi(report, fnames, raws, p, subj=None):
    t0 = time.time()
    section = 'Good HPI count'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    figs = list()
    captions = list()
    for fname, raw in zip(fnames, raws):
        fit_data = _get_fit_data(fname, raw, p, subj, prefix='      ')
        if fit_data is None:
            print('%s skipped, HPI count data not found (possibly '
                  'no params.*_limit values set?)' % (section,))
            break
        fig = plot_good_coils(fit_data, show=False)
        fig.set_size_inches(10, 2)
        fig.tight_layout()
        figs.append(fig)
        captions.append('%s: %s' % (section, op.split(fname)[-1]))
    report.add_figs_to_section(figs, captions, section,
                               image_format='svg')
    print('%5.1f sec' % ((time.time() - t0),))


def _report_chpi_snr(report, fnames, p=None):
    t0 = time.time()
    section = 'cHPI SNR'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    figs = list()
    captions = list()
    for fname in fnames:
        raw = mne.io.read_raw_fif(fname, allow_maxshield='yes')
        t_window = _get_t_window(p, raw)
        fig = plot_chpi_snr_raw(raw, t_window, show=False,
                                verbose=False)
        fig.set_size_inches(10, 5)
        fig.subplots_adjust(0.1, 0.1, 0.8, 0.95,
                            wspace=0, hspace=0.5)
        figs.append(fig)
        captions.append('%s: %s' % (section, op.split(fname)[-1]))
    report.add_figs_to_section(figs, captions, section,
                               image_format='png')  # svd too slow
    print('%5.1f sec' % ((time.time() - t0),))


def _report_raw_segments(report, raw, lowpass=None):
    section = 'Raw segments'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    times = np.linspace(raw.times[0], raw.times[-1], 12)[1:-1]
    raw_plot = list()
    for t in times:
        this_raw = raw.copy().crop(t - 0.5, t + 0.5)
        this_raw.load_data()
        this_raw._data[:] -= np.mean(this_raw._data, axis=-1,
                                     keepdims=True)
        raw_plot.append(this_raw)
    raw_plot = mne.concatenate_raws(raw_plot)
    for key in ('BAD boundary', 'EDGE boundary'):
        raw_plot.annotations.delete(
            np.where(raw_plot.annotations.description == key)[0])
    new_events = np.linspace(
        0, int(round(10 * raw.info['sfreq'])) - 1, 11).astype(int)
    new_events += raw_plot.first_samp
    new_events = np.array([new_events,
                           np.zeros_like(new_events),
                           np.ones_like(new_events)]).T
    fig = raw_plot.plot(group_by='selection', butterfly=True,
                        events=new_events, lowpass=lowpass)
    fig.axes[0].lines[-1].set_zorder(10)  # events
    fig.axes[0].set(xticks=np.arange(0, len(times)) + 0.5)
    xticklabels = ['%0.1f' % t for t in times]
    fig.axes[0].set(xticklabels=xticklabels)
    fig.axes[0].set(xlabel='Center of 1-second segments')
    fig.axes[0].grid(False)
    for _ in range(len(fig.axes) - 1):
        fig.delaxes(fig.axes[-1])
    fig.set(figheight=(fig.axes[0].get_yticks() != 0).sum(),
            figwidth=12)
    fig.subplots_adjust(0.0, 0.0, 1, 1, 0, 0)
    report.add_figs_to_section(fig, section, section, image_format='png')


def _report_raw_psd(report, raw, raw_pca=None, p=None):
    t0 = time.time()
    section = 'PSD'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    if p is None:
        lp_trans = 10
        lp_cut = 40
    else:
        if p.lp_trans == 'auto':
            lp_trans = 0.25 * p.lp_cut
        else:
            lp_trans = p.lp_trans
        lp_cut = p.lp_cut
    del p
    n_fft = 8192
    fmax = raw.info['lowpass']
    figs = [raw.plot_psd(fmax=fmax, n_fft=n_fft, show=False)]
    captions = ['%s: Raw' % section]
    fmax = lp_cut + 2 * lp_trans
    figs.append(raw.plot_psd(fmax=fmax, n_fft=n_fft, show=False))
    captions.append('%s: Raw (zoomed)' % section)
    if raw_pca is not None:
        figs.append(raw_pca.plot_psd(fmax=fmax, n_fft=n_fft,
                                     show=False))
        captions.append('%s: Processed' % section)
    # shared y limits
    n = len(figs[0].axes) // 2
    for ai, axes in enumerate(list(zip(
            *[f.axes for f in figs]))[:n]):
        ylims = np.array([ax.get_ylim() for ax in axes])
        ylims = [np.min(ylims[:, 0]), np.max(ylims[:, 1])]
        for ax in axes:
            ax.set_ylim(ylims)
            ax.set(title='')
    for fig in figs:
        fig.set_size_inches(8, 8)
        with warnings.catch_warnings(record=True):
            fig.tight_layout()
    report.add_figs_to_section(figs, captions, section,
                               image_format='svg')
    print('%5.1f sec' % ((time.time() - t0),))


def gen_html_report(p, subjects, structurals, run_indices=None):
    """Generate HTML reports."""
    import matplotlib.pyplot as plt
    if run_indices is None:
        run_indices = [None] * len(subjects)
    time_kwargs = dict()
    if 'time_unit' in mne.fixes._get_args(mne.viz.plot_evoked):
        time_kwargs['time_unit'] = 's'
    for si, subj in enumerate(subjects):
        struc = structurals[si]
        report = Report(verbose=False)
        print('  Processing subject %s/%s (%s)'
              % (si + 1, len(subjects), subj))

        # raw
        fnames = get_raw_fnames(p, subj, 'raw', erm=False, add_splits=False,
                                run_indices=run_indices[si])
        for fname in fnames:
            if not op.isfile(fname):
                raise RuntimeError('Cannot create reports until raw data '
                                   'exist, missing:\n%s' % fname)
        raw = [_read_raw_prebad(p, subj, fname, False) for fname in fnames]
        _fix_raw_eog_cals(raw, 'all')
        raw = mne.concatenate_raws(raw)

        # sss
        sss_fnames = get_raw_fnames(p, subj, 'sss', False, False,
                                    run_indices[si])
        has_sss = all(op.isfile(fname) for fname in sss_fnames)
        sss_info = mne.io.read_raw_fif(sss_fnames[0]) if has_sss else None
        bad_file = get_bad_fname(p, subj)
        if bad_file is not None:
            sss_info.load_bad_channels(bad_file)
        if sss_info is not None:
            sss_info = sss_info.info

        # pca
        pca_fnames = get_raw_fnames(p, subj, 'pca', False, False,
                                    run_indices[si])
        if all(op.isfile(fname) for fname in pca_fnames):
            raw_pca = [mne.io.read_raw_fif(fname) for fname in pca_fnames]
            _fix_raw_eog_cals(raw_pca, 'all')
            raw_pca = mne.concatenate_raws(raw_pca)
        else:
            raw_pca = None

        # epochs
        epochs_fname, _ = get_epochs_evokeds_fnames(p, subj, p.analyses)
        _, epochs_fname = epochs_fname
        has_epochs = op.isfile(epochs_fname)

        # whitening and source localization
        inv_dir = op.join(p.work_dir, subj, p.inverse_dir)

        has_fwd = op.isfile(op.join(p.work_dir, subj, p.forward_dir,
                                    subj + p.inv_tag + '-fwd.fif'))

        with report_context():
            #
            # Head coils
            #
            if p.report_params.get('good_hpi_count', True) and p.movecomp:
                _report_good_hpi(report, fnames, [None] * len(fnames), p, subj)
            else:
                print('    HPI count skipped')

            #
            # cHPI SNR
            #
            if p.report_params.get('chpi_snr', True) and p.movecomp:
                _report_chpi_snr(report, fnames, p)
            else:
                print('    cHPI SNR skipped')

            #
            # Head movement
            #
            section = 'Head movement'
            if p.report_params.get('head_movement', True) and p.movecomp:
                print(('    %s ... ' % section).ljust(LJUST), end='')
                t0 = time.time()
                trans_to = _load_trans_to(p, subj, run_indices[si], raw)
                figs = list()
                captions = list()
                for fname in fnames:
                    pos, _, _ = _head_pos_annot(
                        p, subj, fname, prefix='      ')
                    fig = plot_head_positions(pos=pos, destination=trans_to,
                                              info=raw.info, show=False)
                    for ax in fig.axes[::2]:
                        """
                        # tighten to the sensor limits
                        assert ax.lines[0].get_color() == (0., 0., 0., 1.)
                        mn, mx = np.inf, -np.inf
                        for line in ax.lines:
                            ydata = line.get_ydata()
                            if np.isfinite(ydata).any():
                                mn = min(np.nanmin(ydata), mn)
                                mx = max(np.nanmax(line.get_ydata()), mx)
                        """
                        # always show at least 10cm span, and use tight limits
                        # if greater than that
                        coord = ax.lines[0].get_ydata()
                        for line in ax.lines:
                            if line.get_color() == 'r':
                                extra = line.get_ydata()[0]
                        mn, mx = coord.min(), coord.max()
                        md = (mn + mx) / 2.
                        mn = min([mn, md - 50., extra])
                        mx = max([mx, md + 50., extra])
                        assert (mn <= coord).all()
                        assert (mx >= coord).all()
                        ax.set_ylim(mn, mx)
                    fig.set_size_inches(10, 6)
                    fig.tight_layout()
                    figs.append(fig)
                    captions.append('%s: %s' % (section, op.split(fname)[-1]))
                del trans_to
                report.add_figs_to_section(figs, captions, section,
                                           image_format='svg')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # Raw segments
            #
            if p.report_params.get('raw_segments', True) and \
                    raw_pca is not None:
                _report_raw_segments(report, raw_pca)

            #
            # PSD
            #
            if p.report_params.get('psd', True):
                _report_raw_psd(report, raw, raw_pca, p)
            else:
                print('    PSD skipped' % section)

            #
            # SSP
            #
            section = 'SSP topomaps'

            proj_nums = _handle_dict(p.proj_nums, subj)
            if p.report_params.get('ssp_topomaps', True) and \
                    raw_pca is not None and np.sum(proj_nums) > 0:
                assert sss_info is not None
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                figs = []
                comments = []
                proj_files = get_proj_fnames(p, subj)
                if p.proj_extra is not None:
                    comments.append('Custom')
                    projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                              p.proj_extra))
                    figs.append(plot_projs_topomap(projs, info=sss_info,
                                                   show=False))
                if any(proj_nums[0]):  # ECG
                    if 'preproc_ecg-proj.fif' in proj_files:
                        comments.append('ECG')
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_ecg-proj.fif'), sss_info,
                            proj_nums[0], p.proj_meg, 'ECG'))
                if any(proj_nums[1]):  # EOG
                    if 'preproc_blink-proj.fif' in proj_files:
                        comments.append('Blink')
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_blink-proj.fif'), sss_info,
                            proj_nums[1], p.proj_meg, 'EOG'))
                if any(proj_nums[2]):  # ERM
                    if 'preproc_cont-proj.fif' in proj_files:
                        comments.append('Continuous')
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_cont-proj.fif'), sss_info,
                            proj_nums[2], p.proj_meg, 'ERM'))
                captions = ['SSP epochs: %s' % c for c in comments]
                report.add_figs_to_section(
                    figs, captions, section, image_format='svg',
                    comments=comments)
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # Source alignment
            #
            section = 'Source alignment'
            source_alignment = p.report_params.get('source_alignment', True)
            if source_alignment is True or isinstance(source_alignment, dict) \
                    and has_sss and has_fwd:
                assert sss_info is not None
                kwargs = source_alignment
                if isinstance(source_alignment, dict):
                    kwargs = dict(**source_alignment)
                else:
                    assert source_alignment is True
                    kwargs = dict()
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                captions = [section]
                try:
                    from mayavi import mlab
                except ImportError:
                    warnings.warn('Cannot plot alignment in Report, mayavi '
                                  'could not be imported')
                else:
                    subjects_dir = mne.utils.get_subjects_dir(
                        p.subjects_dir, raise_error=True)
                    bem, src, trans, _ = _get_bem_src_trans(
                        p, sss_info, subj, struc)
                    if len(mne.pick_types(sss_info)):
                        coord_frame = 'meg'
                    else:
                        coord_frame = 'head'
                    with mlab_offscreen():
                        fig = mlab.figure(bgcolor=(0., 0., 0.),
                                          size=(1000, 1000))
                        for key, val in (
                                ('info', sss_info),
                                ('subjects_dir', subjects_dir), ('bem', bem),
                                ('dig', True), ('coord_frame', coord_frame),
                                ('show_axes', True), ('fig', fig),
                                ('trans', trans), ('src', src)):
                            kwargs[key] = kwargs.get(key, val)
                        try_surfs = [('head-dense', 'inner_skull'),
                                     ('head', 'inner_skull'),
                                     'head',
                                     'inner_skull']
                        for surf in try_surfs:
                            try:
                                mne.viz.plot_alignment(surfaces=surf, **kwargs)
                            except Exception:
                                pass
                            else:
                                break
                        else:
                            raise RuntimeError('Could not plot any surface '
                                               'for alignment:\n%s'
                                               % (try_surfs,))
                        fig.scene.parallel_projection = True
                        view = list()
                        for ai, angle in enumerate([180, 90, 0]):
                            mlab.view(angle, 90, focalpoint=(0., 0., 0.),
                                      distance=0.6, figure=fig)
                            view.append(mlab.screenshot(figure=fig))
                        mlab.close(fig)
                    view = trim_bg(np.concatenate(view, axis=1), 0)
                    report.add_figs_to_section(view, captions, section)
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)
            #
            # Drop log
            #
            section = 'Drop log'
            if p.report_params.get('drop_log', True) and has_epochs:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                epo = read_epochs(epochs_fname)
                figs = [epo.plot_drop_log(subject=subj, show=False)]
                captions = [repr(epo)]
                report.add_figs_to_section(figs, captions, section,
                                           image_format='svg')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # SNR
            #
            section = 'SNR'
            if p.report_params.get('snr', None) is not None:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                snrs = p.report_params['snr']
                if not isinstance(snrs, (list, tuple)):
                    snrs = [snrs]
                for snr in snrs:
                    assert isinstance(snr, dict)
                    analysis = snr['analysis']
                    name = snr['name']
                    times = snr.get('times', [0.1])
                    inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
                    fname_inv = op.join(inv_dir,
                                        safe_inserter(snr['inv'], subj))
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if not op.isfile(fname_inv):
                        print('    Missing inv: %s'
                              % op.basename(fname_inv), end='')
                    elif not op.isfile(fname_evoked):
                        print('    Missing evoked: %s'
                              % op.basename(fname_evoked), end='')
                    else:
                        inv = mne.minimum_norm.read_inverse_operator(fname_inv)
                        this_evoked = mne.read_evokeds(fname_evoked, name)
                        figs = plot_snr_estimate(
                            this_evoked, inv, verbose='error')
                        figs.axes[0].set_ylim(auto=True)
                        captions = ('%s: %s["%s"] (N=%d)'
                                    % (section, analysis, name,
                                       this_evoked.nave))
                        report.add_figs_to_section(
                            figs, captions, section=section,
                            image_format='svg')
                print('%5.1f sec' % ((time.time() - t0),))
            #
            # BEM
            #
            section = 'BEM'
            if p.report_params.get('bem', True) and has_fwd:
                caption = '%s: %s' % (section, struc)
                bem, src, trans, _ = _get_bem_src_trans(
                    p, raw.info, subj, struc)
                if not bem['is_sphere']:
                    subjects_dir = mne.utils.get_subjects_dir(
                        p.subjects_dir, raise_error=True)
                    mri_fname = op.join(subjects_dir, struc, 'mri', 'T1.mgz')
                    if not op.isfile(mri_fname):
                        warnings.warn(
                            'Could not find MRI:\n%s\nIf using surrogate '
                            'subjects, use '
                            'params.report_params["bem"] = False to avoid '
                            'this warning', stacklevel=2)
                    else:
                        t0 = time.time()
                        print(('    %s ... ' % section).ljust(LJUST), end='')
                        report.add_bem_to_section(struc, caption, section,
                                                  decim=10, n_jobs=1,
                                                  subjects_dir=subjects_dir)
                        print('%5.1f sec' % ((time.time() - t0),))
                else:
                    print('    %s skipped (sphere)' % section)
            else:
                print('    %s skipped' % section)

            #
            # Whitening
            #
            section = 'Covariance'
            if p.report_params.get('covariance', True):
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                cov_name = _get_cov_name(p, subj)
                if cov_name is None:
                    print('    Missing covariance: %s'
                          % op.basename(cov_name), end='')
                else:
                    noise_cov = mne.read_cov(cov_name)
                    info = raw_pca.info
                    figs = plot_cov(
                        noise_cov, info, show=False, verbose='error')
                    captions = ['%s: %s' % (section, kind)
                                for kind in ('images', 'SVDs')]
                    report.add_figs_to_section(
                        figs, captions, section=section, image_format='png')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            section = 'Whitening'
            if p.report_params.get('whitening', False):
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')

                whitenings = p.report_params['whitening']
                if not isinstance(whitenings, (list, tuple)):
                    whitenings = [whitenings]
                for whitening in whitenings:
                    assert isinstance(whitening, dict)
                    analysis = whitening['analysis']
                    name = whitening['name']
                    cov_name = _get_cov_name(p, subj, whitening.get('cov'))
                    # Load the inverse
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if cov_name is None:
                        if whitening.get('cov') is not None:
                            extra = ': %s' % op.basename(whitening['cov'])
                        else:
                            extra = ''
                        print('    Missing cov%s' % extra, end='')
                    elif not op.isfile(fname_evoked):
                        print('    Missing evoked: %s'
                              % op.basename(fname_evoked), end='')
                    else:
                        noise_cov = mne.read_cov(cov_name)
                        evo = mne.read_evokeds(fname_evoked, name)
                        captions = ('%s: %s["%s"] (N=%d)'
                                    % (section, analysis, name, evo.nave))
                        fig = evo.plot_white(noise_cov, verbose='error',
                                             **time_kwargs)
                        report.add_figs_to_section(
                            fig, captions, section=section, image_format='png')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # Sensor space plots
            #
            section = 'Responses'
            if p.report_params.get('sensor', False):
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                sensors = p.report_params['sensor']
                if not isinstance(sensors, (list, tuple)):
                    sensors = [sensors]
                for sensor in sensors:
                    assert isinstance(sensor, dict)
                    analysis = sensor['analysis']
                    name = sensor['name']
                    times = sensor.get('times', [0.1, 0.2])
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if not op.isfile(fname_evoked):
                        print('    Missing evoked: %s'
                              % op.basename(fname_evoked), end='')
                    else:
                        this_evoked = mne.read_evokeds(fname_evoked, name)
                        figs = this_evoked.plot_joint(
                            times, show=False, ts_args=dict(**time_kwargs),
                            topomap_args=dict(outlines='head', **time_kwargs))
                        if not isinstance(figs, (list, tuple)):
                            figs = [figs]
                        captions = ('%s: %s["%s"] (N=%d)'
                                    % (section, analysis, name,
                                       this_evoked.nave))
                        captions = [captions] * len(figs)
                        report.add_figs_to_section(
                            figs, captions, section=section,
                            image_format='png')
                print('%5.1f sec' % ((time.time() - t0),))

            #
            # Source estimation
            #
            section = 'Source estimation'
            if p.report_params.get('source', False):
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                sources = p.report_params['source']
                if not isinstance(sources, (list, tuple)):
                    sources = [sources]
                for source in sources:
                    assert isinstance(source, dict)
                    analysis = source['analysis']
                    name = source['name']
                    times = source.get('times', [0.1, 0.2])
                    # Load the inverse
                    inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
                    fname_inv = op.join(inv_dir,
                                        safe_inserter(source['inv'], subj))
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if not op.isfile(fname_inv):
                        print('    Missing inv: %s'
                              % op.basename(fname_inv), end='')
                    elif not op.isfile(fname_evoked):
                        print('    Missing evoked: %s'
                              % op.basename(fname_evoked), end='')
                    else:
                        inv = mne.minimum_norm.read_inverse_operator(fname_inv)
                        this_evoked = mne.read_evokeds(fname_evoked, name)
                        title = ('%s: %s["%s"] (N=%d)'
                                 % (section, analysis, name, this_evoked.nave))
                        stc = mne.minimum_norm.apply_inverse(
                            this_evoked, inv,
                            lambda2=source.get('lambda2', 1. / 9.),
                            method=source.get('method', 'dSPM'))
                        stc = abs(stc)
                        # get clim using the reject_tmin <->reject_tmax
                        stc_crop = stc.copy().crop(
                            p.reject_tmin, p.reject_tmax)
                        clim = source.get('clim', dict(kind='percent',
                                                       lims=[82, 90, 98]))
                        out = mne.viz._3d._limits_to_control_points(
                            clim, stc_crop.data, 'viridis',
                            transparent=True)  # dummy cmap
                        if isinstance(out[0], (list, tuple, np.ndarray)):
                            clim = out[0]  # old MNE
                        else:
                            clim = out[1]  # new MNE (0.17+)
                        clim = dict(kind='value', lims=clim)
                        assert isinstance(stc, (mne.SourceEstimate,
                                                mne.VolSourceEstimate))
                        bem, _, _, _ = _get_bem_src_trans(
                            p, raw.info, subj, struc)
                        is_usable = (isinstance(stc, mne.SourceEstimate) or
                                     not bem['is_sphere'])
                        if not is_usable:
                            print('Only source estimates with individual '
                                  'anatomy supported')
                            break
                        subjects_dir = mne.utils.get_subjects_dir(
                            p.subjects_dir, raise_error=True)
                        kwargs = dict(
                            colormap=source.get('colormap', 'viridis'),
                            transparent=source.get('transparent', True),
                            clim=clim, subjects_dir=subjects_dir)
                        imgs = list()
                        size = source.get('size', (800, 600))
                        if isinstance(stc, mne.SourceEstimate):
                            with mlab_offscreen():
                                brain = stc.plot(
                                    hemi=source.get('hemi', 'split'),
                                    views=source.get('views', ['lat', 'med']),
                                    size=size,
                                    foreground='k', background='w',
                                    **kwargs)
                                for t in times:
                                    brain.set_time(t)
                                    imgs.append(
                                        trim_bg(brain.screenshot(), 255))
                                brain.close()
                        else:
                            # XXX eventually plot_volume_source_estimtates
                            # will have an intial_time arg...
                            mode = source.get('mode', 'stat_map')
                            for t in times:
                                fig = stc.copy().crop(t, t).plot(
                                    src=inv['src'], mode=mode, show=False,
                                    **kwargs,
                                )
                                fig.set_dpi(100.)
                                fig.set_size_inches(*(np.array(size) / 100.))
                                imgs.append(fig)
                        captions = ['%2.3f sec' % t for t in times]
                        report.add_slider_to_section(
                            imgs, captions=captions, section=section,
                            title=title, image_format='png')
                        plt.close('all')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

        report_fname = get_report_fnames(p, subj)[0]
        report.save(report_fname, open_browser=False, overwrite=True)


def _proj_fig(fname, info, proj_nums, proj_meg, kind):
    import matplotlib.pyplot as plt
    proj_nums = np.array(proj_nums, int)
    assert proj_nums.shape == (3,)
    projs = read_proj(fname)
    epochs = fname.replace('-proj.fif', '-epo.fif')
    n_col = proj_nums.max()
    rs_topo = 3
    if op.isfile(epochs):
        epochs = mne.read_epochs(epochs)
        evoked = epochs.average()
        rs_trace = 2
    else:
        rs_trace = 0
    n_row = proj_nums.astype(bool).sum() * (rs_topo + rs_trace)
    shape = (n_row, n_col)
    fig = plt.figure(figsize=(n_col * 2, n_row * 0.75))
    used = np.zeros(len(projs), int)
    ri = 0
    for count, ch_type in zip(proj_nums, ('grad', 'mag', 'eeg')):
        if count == 0:
            continue
        if ch_type == 'eeg':
            meg, eeg = False, True
        else:
            meg, eeg = ch_type, False
        ch_names = [info['ch_names'][pick]
                    for pick in mne.pick_types(info, meg=meg, eeg=eeg)]
        idx = np.where([np.in1d(ch_names, proj['data']['col_names']).all()
                        for proj in projs])[0]
        if len(idx) != count:
            raise RuntimeError('Expected %d %s projector%s for channel type '
                               '%s based on proj_nums but got %d in %s'
                               % (count, kind, _pl(count), ch_type, len(idx),
                                  fname))
        if proj_meg == 'separate':
            assert not used[idx].any()
        else:
            assert (used[idx] <= 1).all()
        used[idx] += 1
        these_projs = [deepcopy(projs[ii]) for ii in idx]
        for proj in these_projs:
            sub_idx = [proj['data']['col_names'].index(name)
                       for name in ch_names]
            proj['data']['data'] = proj['data']['data'][:, sub_idx]
            proj['data']['col_names'] = ch_names
        topo_axes = [plt.subplot2grid(
            shape, (ri * (rs_topo + rs_trace), ci),
            rowspan=rs_topo) for ci in range(count)]
        # topomaps
        with warnings.catch_warnings(record=True):
            plot_projs_topomap(these_projs, info=info, show=False,
                               axes=topo_axes)
        plt.setp(topo_axes, title='', xlabel='')
        topo_axes[0].set(ylabel=ch_type)
        if rs_trace:
            trace_axes = [plt.subplot2grid(
                shape, (ri * (rs_topo + rs_trace) + rs_topo, ci),
                rowspan=rs_trace) for ci in range(count)]
            for proj, ax in zip(these_projs, trace_axes):
                this_evoked = evoked.copy().pick_channels(ch_names)
                p = proj['data']['data']
                assert p.shape == (1, len(this_evoked.data))
                with warnings.catch_warnings(record=True):  # tight_layout
                    this_evoked.plot(
                        picks=np.arange(len(this_evoked.data)), axes=[ax])
                ax.texts = []
                trace = np.dot(p, this_evoked.data)[0]
                trace *= 0.8 * (np.abs(ax.get_ylim()).max() /
                                np.abs(trace).max())
                ax.plot(this_evoked.times, trace, color='#9467bd')
                ax.set(title='', ylabel='', xlabel='')
        ri += 1
    assert used.all() and (used <= 2).all()
    fig.subplots_adjust(0.1, 0.1, 0.95, 1, 0.3, 0.3)
    return fig


def _get_cov_name(p, subj, cov_name=None):
    # just the first for now
    if cov_name is None:
        if p.inv_names:
            cov_name = (safe_inserter(p.inv_names[0], subj) +
                        ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
        elif p.runs_empty:  # erm cov
            new_run = safe_inserter(p.runs_empty[0], subj)
            cov_name = new_run + p.pca_extra + p.inv_tag + '-cov.fif'
    if cov_name is not None:
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        cov_name = op.join(cov_dir, cov_name)
        if not op.isfile(cov_name):
            cov_name = None
    return cov_name
