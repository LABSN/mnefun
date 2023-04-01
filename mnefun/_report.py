#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Create HTML reports."""

from contextlib import contextmanager
from copy import deepcopy
import os.path as op
import time
import warnings

import numpy as np
from scipy.signal import find_peaks, peak_prominences

import mne
from mne import read_proj, read_epochs, find_events, pick_types
from mne.io import BaseRaw
from mne.viz import (plot_projs_topomap, plot_cov, plot_snr_estimate,
                     plot_events)
from mne.viz._3d import plot_head_positions
from mne.report import Report
from mne.utils import _pl, use_log_level
from mne.cov import whiten_evoked
from mne.viz.utils import _triage_rank_sss

from ._epoching import _concat_resamp_raws
from ._forward import _get_bem_src_trans
from ._paths import (get_raw_fnames, get_proj_fnames, get_report_fnames,
                     get_bad_fname, get_epochs_evokeds_fnames, safe_inserter)
from ._ssp import _proj_nums
from ._sss import (_load_trans_to, _read_raw_prebad,
                   _get_t_window, _get_fit_data)
from ._viz import plot_good_coils, plot_chpi_snr_raw, trim_bg, mlab_offscreen
from ._utils import _handle_dict

LJUST = 25
SQRT_2 = np.sqrt(2)
SQ2STR = '×√2'


@contextmanager
def report_context():
    """Create a context for making plt and mlab figures."""
    mne.viz.set_browser_backend('matplotlib')
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


# Backward compat wrappers for MNE 1.0+
def _add_figs_to_section(report, figs, captions, section, image_format='png'):
    try:
        report.add_figure
    except AttributeError:
        report.add_figs_to_section(figs, captions, section,
                                   image_format=image_format)
    else:
        # MNE 1.0+
        if isinstance(captions, (list, tuple)):
            assert isinstance(figs, (list, tuple))
            for f, c in zip(figs, captions):
                report.add_figure(
                    f, title=section, caption=c,
                    image_format=image_format,
                    tags=(section.replace(' ', '_'),))
        else:
            assert not isinstance(figs, (list, tuple))
            report.add_figure(
                figs, title=section, caption=captions,
                image_format=image_format,
                tags=(section.replace(' ', '_'),))


def _add_slider_to_section(report, figs, captions, section, title,
                           image_format='png'):
    try:
        report.add_figure
    except AttributeError:
        report.add_slider_to_section(
            report, figs, captions=captions, section=section, title=title,
            image_format=image_format)
    else:
        report.add_figure(
            figs, title=title, caption=captions, image_format=image_format,
            tags=(section.replace(' ', '_'),))


def _add_bem_to_section(report, struc, caption, section, **kwargs):
    try:
        report.add_bem
    except AttributeError:
        report.add_bem_to_section(struc, caption, section, **kwargs)
    else:
        report.add_bem(struc, title=section, **kwargs)  # caption not used


def _check_fname_raw(fname, p, subj):
    if isinstance(fname, str):
        raw = _read_raw_prebad(p, subj, fname, disp=False)
    else:
        assert isinstance(fname, BaseRaw)
        fname, raw = fname.filenames[0], fname
    return fname, raw


def _report_good_hpi(report, fnames, p=None, subj=None):
    t0 = time.time()
    section = 'Good HPI count'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    figs = list()
    captions = list()
    for fname in fnames:
        fname, raw = _check_fname_raw(fname, p, subj)
        fit_data, _, _ = _get_fit_data(raw, p, prefix='      ')
        if fit_data is None:
            print('%s skipped, HPI count data not found (possibly '
                  'no params.*_limit values set?)' % (section,))
            break
        fig = plot_good_coils(fit_data, show=False)
        fig.set_size_inches(10, 2)
        fig.tight_layout()
        figs.append(fig)
        captions.append('%s: %s' % (section, op.basename(fname)))
    _add_figs_to_section(
        report, figs, captions, section, image_format='png')
    print('%5.1f sec' % ((time.time() - t0),))


def _report_chpi_snr(report, fnames, p=None, subj=None):
    t0 = time.time()
    section = 'cHPI SNR'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    figs = list()
    captions = list()
    for fname in fnames:
        fname, raw = _check_fname_raw(fname, p, subj)
        if raw is None:
            raw = _read_raw_prebad(p, subj, fname, disp=False)
        t_window = _get_t_window(p, raw)
        fig = plot_chpi_snr_raw(raw, t_window, show=False,
                                verbose=False)
        fig.set_size_inches(10, 5)
        fig.subplots_adjust(0.1, 0.1, 0.8, 0.95,
                            wspace=0, hspace=0.5)
        figs.append(fig)
        captions.append('%s: %s' % (section, op.basename(fname)))
    _add_figs_to_section(
        report, figs, captions, section, image_format='png')  # svd too slow
    print('%5.1f sec' % ((time.time() - t0),))


def _report_head_movement(report, fnames, p=None, subj=None, run_indices=None):
    section = 'Head movement'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    t0 = time.time()
    figs = list()
    captions = list()
    for fname in fnames:
        fname, raw = _check_fname_raw(fname, p, subj)
        _, pos, _ = _get_fit_data(raw, p, prefix='      ')
        trans_to = _load_trans_to(p, subj, run_indices, raw)
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
        captions.append('%s: %s' % (section, op.basename(fname)))
    del trans_to
    _add_figs_to_section(report, figs, captions, section, image_format='png')
    print('%5.1f sec' % ((time.time() - t0),))


def _report_events(report, fnames, p=None, subj=None):
    t0 = time.time()
    section = 'Events'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    figs = list()
    captions = list()
    for fname in fnames:
        fname, raw = _check_fname_raw(fname, p, subj)
        events = find_events(raw, stim_channel='STI101', shortest_event=1,
                             initial_event=True)
        if len(events) > 0:
            fig = plot_events(events, raw.info['sfreq'], raw.first_samp)
            fig.set_size_inches(10, 4)
            fig.subplots_adjust(0.1, 0.1, 0.9, 0.99, wspace=0, hspace=0)
            figs.append(fig)
            captions.append('%s: %s' % (section, op.basename(fname)))
    if len(figs):
        _add_figs_to_section(
            report, figs, captions, section, image_format='png')
    print('%5.1f sec' % ((time.time() - t0),))


def _report_raw_segments(report, raw, lowpass=None):
    t0 = time.time()
    section = 'Raw segments'
    print(('    %s ... ' % section).ljust(LJUST), end='')
    times = np.linspace(raw.times[0], raw.times[-1], 12)[1:-1]
    raw_plot = list()
    for t in times:
        this_raw = raw.copy().crop(
            max(t - 0.5, 0), min(t + 0.5, raw.times[-1]))
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
    with mne.utils.use_log_level('error'):
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
    fig.subplots_adjust(0.025, 0.0, 1, 1, 0, 0)
    _add_figs_to_section(report, fig, section, section, image_format='png')
    print('%5.1f sec' % ((time.time() - t0),))


def _gen_psd_plot(raw, fmax, n_fft, ax):
    if hasattr(mne.time_frequency.Spectrum, 'plot'):
        plot = raw.compute_psd(fmax=fmax, n_fft=n_fft).plot(show=False,
                                                            axes=ax)
    else:
        plot = raw.plot_psd(fmax=fmax, n_fft=n_fft, show=False, ax=ax)
    return plot


def _report_raw_psd(report, raw, raw_pca=None, raw_erm=None, raw_erm_pca=None,
                    p=None):
    t0 = time.time()
    section = 'PSD'
    import matplotlib.pyplot as plt
    print(('    %s ... ' % section).ljust(LJUST), end='')
    if p.lp_trans == 'auto':
        lp_trans = 0.25 * p.lp_cut
    else:
        lp_trans = p.lp_trans
    lp_cut = p.lp_cut
    del p
    n_fft = min(8192, len(raw.times))
    fmax = raw.info['lowpass']
    n_ax = sum(key in raw for key in ('mag', 'grad', 'eeg'))
    _, ax = plt.subplots(n_ax, figsize=(10, 8))
    figs = [_gen_psd_plot(raw, fmax=fmax, n_fft=n_fft, ax=ax)]
    captions = ['%s: Raw' % section]
    fmax = lp_cut + 2 * lp_trans
    for this_raw, caption in [
            (raw, f'{section}: Raw (zoomed)'),
            (raw_pca, f'{section}: Raw processed (zoomed)'),
            (raw_erm, f'{section}: ERM (zoomed)'),
            (raw_erm_pca, f'{section}: ERM processed (zoomed)')]:
        _, ax = plt.subplots(n_ax, figsize=(10, 8))
        if this_raw is not None:
            figs.append(_gen_psd_plot(this_raw, fmax=fmax, n_fft=n_fft, ax=ax))
            captions.append(caption)
    # shared y limits
    n = len(figs[0].axes) // 2
    for ai, axes in enumerate(list(zip(
            *[f.axes for f in figs]))[:n]):
        ylims = np.array([ax.get_ylim() for ax in axes])
        ylims = [np.min(ylims[:, 0]), np.max(ylims[:, 1])]
        for ax in axes:
            ax.set_ylim(ylims)
            ax.set(title='')
    _add_figs_to_section(
        report, figs, captions, section, image_format='png')
    print('%5.1f sec' % ((time.time() - t0),))


def _get_memo_times(evoked, cov_name, key, memo):
    if cov_name is None:
        raise RuntimeError(
            'A noise covariance must be provided in '
            'sensor["cov"] if times="peaks"')
    cov = mne.read_cov(cov_name)
    if key not in memo:
        memo[key] = _peak_times(evoked, cov)
    return memo[key]


def _peak_times(evoked, cov, max_peaks=5):
    """Return times of prominent peaks from whitened global field power."""
    with use_log_level('error'):
        rank_dict = _triage_rank_sss(evoked.info, [cov])[1][0]
    evoked = whiten_evoked(evoked, cov, rank=rank_dict)
    thr = 1
    # Calculate gfps
    gfp_list = []
    if 'meg' in evoked:
        evk = evoked.copy().pick_types(meg=True)
        gfp = np.sum(evk.data ** 2, axis=0) / rank_dict['meg']
        gfp_list.append(gfp)
    if 'eeg' in evoked:
        evk = evoked.copy().pick_types(meg=False, eeg=True)
        gfp = np.sum(evk.data ** 2, axis=0) / rank_dict['eeg']
        gfp_list.append(gfp)
    if not gfp_list:    # for non-meg/eeg data types
        gfp = np.mean(evk.data ** 2, axis=0)
    else:
        gfp = np.array(gfp_list).sum(axis=0) / len(gfp_list)
    # Find peaks
    npeaks = max(min(len(gfp) // 3, max_peaks), 1)  # npeaks >=1, <= max_peaks
    peaks = find_peaks(gfp, height=thr)[0]
    prms = peak_prominences(gfp, peaks)[0]
    times = peaks[prms.argsort()[::-1]][:npeaks]
    times.sort()
    if not len(times):  # guarantee at least 1
        times = [gfp.argmax()]
    times = evoked.times[times]
    print('    Whitened GFP peak%s (%s):   %s\n                         '
          % (_pl(times), evoked.comment,
             ','.join('%0.3f' % t for t in times),), end='')
    return times


def _get_std_even_odd(fname_evoked, name, proj=True):
    proj = True if proj == 'reconstruct' else proj
    all_evoked = [mne.read_evokeds(fname_evoked, name, proj=proj)]
    for extra in (' even', ' odd'):
        try:
            this_evoked = mne.read_evokeds(
                fname_evoked, name + extra, proj=proj)
        except ValueError:
            pass
        else:
            all_evoked += [this_evoked]
    return all_evoked


def gen_html_report(p, subjects, structurals, run_indices=None):
    """Generate HTML reports."""
    import matplotlib.pyplot as plt
    if run_indices is None:
        run_indices = [None] * len(subjects)
    known_keys = {
        'good_hpi_count', 'chpi_snr', 'head_movement', 'raw_segments', 'psd',
        'ssp_topomaps', 'source_alignment', 'drop_log', 'bem', 'covariance',
        'whitening', 'snr', 'sensor', 'source', 'pre_fun', 'post_fun',
        'preload',
    }
    unknown = set(p.report_params.keys()).difference(known_keys)
    if unknown:
        raise RuntimeError(f'unknown report_params: {sorted(unknown)}')
    preload = p.report_params.get('preload', False)
    for si, subj in enumerate(subjects):
        struc = structurals[si] if structurals is not None else None
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
        raw, _ = _concat_resamp_raws(
            p, subj, fnames, fix='all', prebad=True, preload=preload,
            set_dev_head_t=True)

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

        # pca / erm / erm_pca
        extra_raws = dict()
        for key, which, erm in [
                ('raw_pca', 'pca', False),
                ('raw_erm', 'raw', 'only'),
                ('raw_erm_pca', 'pca', 'only')]:
            these_fnames = get_raw_fnames(
                p, subj, which, erm, False, run_indices[si])
            if len(these_fnames) and all(op.isfile(f) for f in these_fnames):
                extra_raws[key], _ = _concat_resamp_raws(
                    p, subj, these_fnames, 'all', preload=True)
                extra_raws[key].apply_proj()
        raw_pca = extra_raws.get('raw_pca', None)
        raw_erm = extra_raws.get('raw_erm', None)
        raw_erm_pca = extra_raws.get('raw_erm_pca', None)
        del extra_raws

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
            # Custom pre-fun
            #
            pre_fun = p.report_params.get('pre_fun', None)
            if pre_fun is not None:
                print('    Pre fun ...'.ljust(LJUST), end='')
                t0 = time.time()
                pre_fun(report, p, subj)
                print('%5.1f sec' % ((time.time() - t0),))

            #
            # Head coils
            #
            if p.report_params.get('good_hpi_count', True) and p.movecomp:
                _report_good_hpi(report, fnames, p, subj)
            else:
                print('    HPI count skipped')

            #
            # cHPI SNR
            #
            if p.report_params.get('chpi_snr', True) and p.movecomp:
                _report_chpi_snr(report, fnames, p, subj)
            else:
                print('    cHPI SNR skipped')

            #
            # Head movement
            #
            if p.report_params.get('head_movement', True) and p.movecomp:
                _report_head_movement(report, fnames, p, subj, run_indices[si])
            else:
                print('    Head movement skipped')

            #
            # Raw segments
            #
            if p.report_params.get('raw_segments', True) and \
                    raw_pca is not None:
                _report_raw_segments(report, raw_pca)
            else:
                print('    Raw segments skipped')

            #
            # PSD
            #
            if p.report_params.get('psd', True):
                _report_raw_psd(report, raw, raw_pca, raw_erm, raw_erm_pca, p)
            else:
                print('    PSD skipped')

            #
            # SSP
            #
            section = 'SSP topomaps'
            proj_nums = _proj_nums(p, subj)
            if p.report_params.get('ssp_topomaps', True) and \
                    raw_pca is not None and np.sum(proj_nums) > 0:
                assert sss_info is not None
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                figs = []
                comments = []
                proj_files = get_proj_fnames(p, subj)
                duration = raw.times[-1]
                if p.proj_extra is not None:
                    comments.append('Custom')
                    projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                              p.proj_extra))
                    figs.append(plot_projs_topomap(projs, info=sss_info,
                                                   show=False))
                if any(proj_nums[2]):  # ERM
                    if 'preproc_cont-proj.fif' in proj_files:
                        if p.cont_as_esss:
                            extra = ' (eSSS)'
                            use_info = raw.info
                        else:
                            extra = ''
                            use_info = sss_info
                        comments.append('Continuous%s' % (extra,))
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_cont-proj.fif'), use_info,
                            proj_nums[2], p.proj_meg, 'ERM', None,
                            duration))
                if any(proj_nums[0]):  # ECG
                    if 'preproc_ecg-proj.fif' in proj_files:
                        ecg_channel = _handle_dict(p.ecg_channel, subj)
                        comments.append('ECG')
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_ecg-proj.fif'), sss_info,
                            proj_nums[0], p.proj_meg, 'ECG', ecg_channel,
                            duration))
                for idx, kind in ((1, 'EOG'), (3, 'HEOG'), (4, 'VEOG')):
                    if any(proj_nums[idx]):  # Blink
                        bk = dict(EOG='Blink').get(kind, kind)
                        if f'preproc_{bk.lower()}-proj.fif' in proj_files:
                            eog_channel = _handle_dict(
                                getattr(p, f'{kind.lower()}_channel'), subj)
                            comments.append(dict(EOG='Blink').get(kind, kind))
                            figs.append(_proj_fig(op.join(
                                p.work_dir, subj, p.pca_dir,
                                f'preproc_{bk.lower()}-proj.fif'), sss_info,
                                proj_nums[idx], p.proj_meg, kind, eog_channel,
                                duration))
                captions = ['SSP epochs: %s' % c for c in comments]
                _add_figs_to_section(
                    report, figs, captions, section, image_format='png')
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
                    mne.viz.get_3d_backend() is not None
                except Exception:
                    warnings.warn('Cannot plot alignment in Report, mayavi '
                                  'could not be imported')
                else:
                    subjects_dir = mne.utils.get_subjects_dir(
                        p.subjects_dir, raise_error=True)
                    bem, src, trans, _ = _get_bem_src_trans(
                        p, sss_info, subj, struc)
                    if len(mne.pick_types(sss_info, meg=True)):
                        coord_frame = 'meg'
                    else:
                        coord_frame = 'head'
                    with mlab_offscreen():
                        fig = mne.viz.create_3d_figure(
                            bgcolor=(0., 0., 0.), size=(1000, 1000))
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
                        ex = None
                        for surf in try_surfs:
                            try:
                                mne.viz.plot_alignment(surfaces=surf, **kwargs)
                            except Exception as exc:
                                ex = exc
                            else:
                                break
                        else:
                            print(
                                '\nCould not plot any surface for alignment '
                                f'for {subj}:\n{try_surfs}\nGot error:\n')
                            raise ex from None
                        del ex
                        try:
                            fig.scene.parallel_projection = True
                        except AttributeError:
                            pass
                        view = list()
                        for ai, angle in enumerate([180, 90, 0]):
                            mne.viz.set_3d_view(
                                fig, angle, 90, focalpoint=(0., 0., 0.),
                                distance=0.6)
                            try:
                                screenshot = fig.plotter.screenshot()
                            except AttributeError:
                                from mayavi import mlab
                                screenshot = mlab.screenshot(fig)
                            view.append(screenshot)
                        try:
                            fig.plotter.close()
                        except AttributeError:
                            from mayavi import mlab
                            mlab.close(fig)
                    view = trim_bg(np.concatenate(view, axis=1), 0)
                    _add_figs_to_section(report, [view], captions, section)
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
                _add_figs_to_section(
                    report, figs, captions, section, image_format='svg')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

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
                        with use_log_level('error'):
                            _add_bem_to_section(
                                report, struc, caption, section, decim=10,
                                n_jobs=1, subjects_dir=subjects_dir)
                        print('%5.1f sec' % ((time.time() - t0),))
                else:
                    print('    %s skipped (sphere)' % section)
            else:
                print('    %s skipped' % section)

            #
            # Covariance
            #
            section = 'Covariance'
            if p.report_params.get('covariance', True):
                t0 = time.time()
                print(('    %s ... ' % section).ljust(LJUST), end='')
                cov_name = p.report_params.get('covariance', None)
                cov_name = _get_cov_name(p, subj, cov_name)
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
                    _add_figs_to_section(
                        report, figs, captions, section=section,
                        image_format='png')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # Whitening
            #
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
                        # too messy to plot separately, just plot the main one
                        all_evoked = _get_std_even_odd(fname_evoked, name)
                        figs = list()
                        n_s = sum(k in all_evoked[0] for k in ('meg', 'eeg'))
                        n_e = len(all_evoked)
                        n_row = n_s * n_e + 1
                        figs, axes = plt.subplots(
                            n_row, 1, figsize=(7, 3 * n_row))
                        captions = [
                            '%s: %s["%s"] (N=%s)'
                            % (section, analysis, all_evoked[0].comment,
                               '/'.join(str(evo.nave) for evo in all_evoked))]
                        for ei, evo in enumerate(all_evoked):
                            if ei > 0:
                                evo.data *= SQRT_2
                            sl = slice(ei, n_e * n_s, n_e)
                            these_axes = list(axes[sl]) + [axes[-1]]
                            evo.plot_white(
                                noise_cov, verbose='error', axes=these_axes)
                            for ax in these_axes[:-1]:
                                n_text = 'N=%d' % (evo.nave,)
                                if ei != 0:
                                    title = f'{n_text} {SQ2STR}'
                                else:
                                    title = f'{ax.get_title()[:-1]}; {n_text})'
                                ax.set(title=title)
                        xlim = all_evoked[0].times[[0, -1]]
                        del ei, all_evoked

                        # joint ylims
                        for si in range(n_s + 1):
                            if si == n_s:
                                ax = axes[-1:]
                            else:
                                ax = axes[si * n_e:(si + 1) * n_e]
                            this_max = max(np.nanmax(np.abs(line.get_ydata()))
                                           for a in ax
                                           for line in a.lines)
                            this_max = 1 if np.isnan(this_max) else this_max
                            if si == n_s:
                                ax[0].set(ylim=[0, this_max], xlim=xlim)
                            else:
                                for a in ax:
                                    a.set(ylim=[-this_max, this_max],
                                          xlim=xlim)
                        del si
                        n_real = 0
                        hs, labels = [], []
                        for line in axes[-1].lines:
                            if line.get_linestyle() == '-':
                                if n_real < n_s:
                                    line.set(linewidth=2)
                                    hs, labels = [line], [line.get_label()]
                                else:
                                    line.set(alpha=0.5, linewidth=1)
                                n_real += 1
                        assert n_real == n_e * n_s
                        axes[-1].legend(hs, labels)
                        if n_e > 1:
                            axes[-1].set_title(
                                f'{axes[-1].get_title()} (halves {SQ2STR})')
                        axes[-1]
                        figs.tight_layout()
                        figs = [figs]
                        _add_figs_to_section(
                            report, figs, captions, section=section,
                            image_format='png')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # SNR
            #
            section = 'SNR'
            if p.report_params.get('snr', None) not in [None, False]:
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
                        figs, ax = plt.subplots(1, figsize=(7, 5))
                        all_evoked = _get_std_even_odd(fname_evoked, name)
                        for ei, evoked in enumerate(all_evoked):
                            if ei != 0:
                                evoked.data *= SQRT_2
                            orig = evoked.nave
                            try:
                                evoked.nave = max(orig, 1)
                                plot_snr_estimate(
                                    evoked, inv, axes=ax, verbose='error')
                            finally:
                                evoked.nave = orig
                        if len(all_evoked) > 1:
                            ax.set_title(
                                f'{all_evoked[0].comment} (halves {SQ2STR})')
                        n_real = 0
                        for line in ax.lines:
                            # some are :'s at zero x/y
                            if line.get_linestyle() == ':':
                                if n_real >= 2:
                                    line.set_visible(False)
                            else:
                                if n_real < 2:
                                    line.set(linewidth=2)
                                else:
                                    line.set(linewidth=1, alpha=0.5)
                                n_real += 1
                        assert n_real == 2 * len(all_evoked)
                        ax.set(
                            ylim=[0, max(np.max(line.get_ydata())
                                         for line in ax.lines)])
                        captions = ('%s: %s["%s"] (N=%s)'
                                    % (section, analysis, name,
                                       '/'.join(str(e.nave)
                                                for e in all_evoked)))
                        figs.tight_layout()
                        _add_figs_to_section(
                            report, figs, captions, section=section,
                            image_format='png')
                print('%5.1f sec' % ((time.time() - t0),))

            #
            # Sensor space plots
            #
            section = 'Responses'
            times_memo = dict()     # store for source plots
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
                    proj = sensor.get('proj', True)
                    assert proj in (True, False, 'reconstruct')
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    bn = op.basename(fname_evoked)
                    if not op.isfile(fname_evoked):
                        print(f'    Missing evoked: {bn}', end='')
                        continue
                    try:
                        this_evoked = mne.read_evokeds(fname_evoked, name)
                    except ValueError as exc:
                        a = exc
                        if 'names in FIF' in str(exc):
                            print(f'    Evoked has no conditions: {bn}')
                            continue
                        raise
                    # Define the time slices to include
                    times = sensor.get('times', [0.1, 0.2])
                    if isinstance(times, str) and times == 'peaks':
                        cov_name = _get_cov_name(p, subj, sensor.get('cov'))
                        times = _get_memo_times(
                            this_evoked, cov_name, (fname_evoked, name),
                            times_memo)
                    del this_evoked
                    # Plot the responses, including even/odd
                    all_evoked = _get_std_even_odd(
                        fname_evoked, name, proj=proj)
                    for this_evoked in all_evoked:
                        if this_evoked.proj and not proj:
                            raise RuntimeError(
                                'cannot use proj=False unless '
                                'p.epochs_proj = "delayed" was run')
                    if proj == 'reconstruct':
                        assert proj in this_evoked.plot.__doc__, \
                            'MNE >= PR #8033 required'
                    all_figs, all_captions = list(), list()
                    for key in ('grad', 'mag', 'eeg'):
                        if key not in all_evoked[0]:
                            continue
                        if key == 'eeg':
                            kwargs = dict(meg=False, eeg=True)
                        else:
                            kwargs = dict(meg=key, eeg=False)
                        picks = pick_types(
                            all_evoked[0].info, exclude='bads', **kwargs)
                        max_ = max(
                            np.abs(e.data[picks]).max() for e in all_evoked)
                        max_ = max_ * mne.defaults.DEFAULTS['scalings'][key]
                        min_ = -max_ if key != 'grad' else 0
                        cmap = 'RdBu_r' if key != 'grad' else 'Reds'
                        for this_evoked in all_evoked:
                            n_text = f'N={this_evoked.nave}'
                            # Always view EEG data with avg ref applied
                            if key == 'eeg' and proj in ('reconstruct', False):
                                this_evoked = this_evoked.copy()
                                all_proj = this_evoked.info['projs']
                                for pr in this_evoked.info['projs']:
                                    pr['active'] = False
                                this_evoked.del_proj()
                                this_evoked.set_eeg_reference(projection=True)
                                this_evoked.apply_proj()
                                this_evoked.add_proj(all_proj)
                            if this_evoked.nave > 0:
                                with mne.utils.use_log_level('error'):
                                    fig = this_evoked.plot_joint(
                                        times, show=False, picks=picks,
                                        ts_args=dict(proj=proj),
                                        topomap_args=dict(
                                            outlines='head',
                                            vmin=min_, vmax=max_,
                                            cmap=cmap, proj=proj))
                                assert isinstance(fig, plt.Figure)
                                fig.axes[0].set(ylim=(-max_, max_))
                                t = fig.axes[-1].texts[0]
                                t.set_text(
                                    f'{t.get_text()}; {n_text})')
                            else:
                                fig = plt.figure()
                            all_figs += [fig]
                            all_captions += [n_text]
                    title = f'{section}: {analysis}["{all_evoked[0].comment}"]'
                    if not proj:
                        title += ' : SSP off'
                    elif proj == 'reconstruct':
                        title += ' : SSP+recon'
                    else:
                        title += ' : SSP on'
                    _add_slider_to_section(
                        report, all_figs, all_captions, section=section,
                        title=title, image_format='png')
                    del this_evoked, all_evoked
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
                    # Load the necessary data
                    inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
                    fname_inv = op.join(inv_dir,
                                        safe_inserter(source['inv'], subj))
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if not op.isfile(fname_inv):
                        print('    Missing inv: %s'
                              % op.basename(fname_inv), end='')
                        continue
                    elif not op.isfile(fname_evoked):
                        print('    Missing evoked: %s'
                              % op.basename(fname_evoked), end='')
                        continue
                    # Generate the STC
                    inv = mne.minimum_norm.read_inverse_operator(fname_inv)
                    this_evoked = mne.read_evokeds(fname_evoked, name)
                    this_evoked.nave = max(this_evoked.nave, 1)
                    stc = mne.minimum_norm.apply_inverse(
                        this_evoked, inv,
                        lambda2=source.get('lambda2', 1. / 9.),
                        method=source.get('method', 'dSPM'))
                    stc = abs(stc)
                    # get clim using the reject_tmin <-> reject_tmax
                    stc_crop = stc.copy().crop(
                        p.reject_tmin, p.reject_tmax)
                    clim = source.get('clim', dict(kind='percent',
                                                   lims=[82, 90, 98]))
                    try:
                        func = mne.viz._3d._limits_to_control_points
                    except AttributeError:  # 0.20+
                        clim = mne.viz._3d._process_clim(
                            clim, 'viridis', transparent=True,
                            data=stc_crop.data)['clim']
                    else:
                        out = func(
                            clim, stc_crop.data, 'viridis',
                            transparent=True)  # dummy cmap
                        if isinstance(out[0], (list, tuple, np.ndarray)):
                            clim = out[0]  # old MNE
                        else:
                            clim = out[1]  # new MNE (0.17+)
                        del out
                        clim = dict(kind='value', lims=clim)
                    if not np.isfinite(clim.get('lims', [0])).all():
                        clim['lims'] = [0, 1, 2]
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
                    size = source.get('size', (800, 600))
                    # Define the time slices to include
                    times = source.get('times', [0.1, 0.2])
                    if isinstance(times, str) and times == 'peaks':
                        cov_name = _get_cov_name(
                            p, subj, source.get('cov'))
                        times = _get_memo_times(
                            this_evoked, cov_name, (fname_evoked, name),
                            times_memo)
                    # Create the STC plots
                    hemi = source.get('hemi', 'split')
                    views = source.get('views', ['lat', 'med'])
                    method = source.get('method', 'dSPM')
                    lambda2 = source.get('lambda2', 1. / 9.)
                    all_evoked = _get_std_even_odd(fname_evoked, name)
                    for ei, this_evoked in enumerate(all_evoked):
                        # change from original to halves
                        if ei == 1 and method in ('dSPM', 'sLORETA'):
                            clim['lims'] = np.array(clim['lims']) / SQRT_2
                        figs = list()
                        orig = this_evoked.nave
                        try:
                            this_evoked.nave = max(this_evoked.nave, 1)
                            stc = mne.minimum_norm.apply_inverse(
                                this_evoked, inv, lambda2=lambda2,
                                method=method)
                        finally:
                            this_evoked.nave = orig
                        stc = abs(stc)
                        if isinstance(stc, mne.SourceEstimate):
                            with mlab_offscreen():
                                brain = stc.plot(
                                    hemi=hemi, views=views, size=size,
                                    foreground='k', background='w',
                                    time_viewer=False, show_traces=False,
                                    **kwargs)
                                for t in times:
                                    try:
                                        brain.set_time(t)
                                    except AttributeError:
                                        idx = np.argmin(np.abs(t - stc.times))
                                        brain.set_time_point(idx)
                                    figs.append(
                                        trim_bg(brain.screenshot(), 255))
                                brain.close()
                        else:
                            mode = source.get('mode', 'glass_brain')
                            for t in times:
                                fig = stc.copy().plot(
                                    src=inv['src'], mode=mode, show=False,
                                    initial_time=t,
                                    **kwargs,
                                )
                                fig.set_dpi(100.)
                                fig.set_size_inches(*(np.array(size) / 100.))
                                figs.append(fig)
                        extra = '' if ei == 0 else SQ2STR
                        title = ('%s: %s["%s"]%s (N=%d)'
                                 % (section, analysis, name, extra,
                                    this_evoked.nave,))
                        captions = ['%2.3f sec' % t for t in times]
                        print(f'add {repr(title)}')
                        print(repr(captions))
                        _add_slider_to_section(
                            report, figs, captions=captions, section=section,
                            title=title, image_format='png')

                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)
            plt.close('all')

            #
            # Custom post-fun
            #
            post_fun = p.report_params.get('post_fun', None)
            if post_fun is not None:
                print('    Post fun ...'.ljust(LJUST), end='')
                t0 = time.time()
                post_fun(report, p, subj)
                print('%5.1f sec' % ((time.time() - t0),))

        report_fname = get_report_fnames(p, subj)[0]
        report.save(report_fname, open_browser=False, overwrite=True)


def _proj_fig(fname, info, proj_nums, proj_meg, kind, use_ch, duration):
    import matplotlib.pyplot as plt
    from mne.preprocessing.ecg import _get_ecg_channel_index
    from mne.preprocessing.eog import _get_eog_channel_index
    proj_nums = np.array(proj_nums, int)
    assert proj_nums.shape == (3,)
    projs = read_proj(fname)
    epochs = fname.replace('-proj.fif', '-epo.fif')
    if op.isfile(epochs):
        epochs = mne.read_epochs(epochs)
        evoked = epochs.average(picks='all')
        title = (
            f'N={len(epochs)}/{len(epochs.drop_log)} '
            f'({len(epochs.drop_log) / duration * 60:0.1f} BPM)'
        )
        cs_trace = 2
        if kind != 'ERM':
            if kind == 'ECG':
                use_ch = [_get_ecg_channel_index(use_ch, evoked)]
            else:
                assert kind in ('EOG', 'HEOG', 'VEOG')
                use_ch = _get_eog_channel_index(use_ch, evoked)
            use_ch = [evoked.ch_names[u] for u in use_ch]
            title += f'\n{", ".join(use_ch)}'
    else:
        cs_trace = 0
    n_col = proj_nums.max() + 2 * cs_trace + 1  # room for legend
    n_row = proj_nums.astype(bool).sum()
    shape = (n_row, n_col)
    # with plt.rc_context({'figure.constrained_layout.use': True}):
    fig = plt.figure(figsize=(n_col * 1.5 + .5, n_row * 2 + .5))
    if cs_trace:
        fig.suptitle(title, fontsize='small')
    used = np.zeros(len(projs), int)
    ri = 0
    colors = ['#CC3311', '#009988', '#0077BB', '#EE3377', '#EE7733', '#33BBEE']
    need_legend = True
    type_titles = dict(eeg='EEG', grad='Grad', mag='Mag')
    last_ax = [None] * 2
    for count, ch_type in zip(proj_nums, ('grad', 'mag', 'eeg')):
        if count == 0:
            continue
        if ch_type == 'eeg':
            meg, eeg = False, True
        else:
            meg, eeg = ch_type, False
        ch_names = [info['ch_names'][pick]
                    for pick in mne.pick_types(info, meg=meg, eeg=eeg)]
        # Some of these will be missing because of prebads
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
        topo_axes = [plt.subplot2grid(shape, (ri, ci + cs_trace))
                     for ci in range(count)]
        # topomaps
        with warnings.catch_warnings(record=True):
            plot_projs_topomap(these_projs, info=info, show=False,
                               axes=topo_axes)
        plt.setp(topo_axes, title='', xlabel='')
        unit = mne.defaults.DEFAULTS['units'][ch_type]
        if cs_trace:
            ax = plt.subplot2grid(shape, (ri, n_col - cs_trace - 1),
                                  colspan=cs_trace)
            this_evoked = evoked.copy().pick_channels(ch_names)
            p = np.concatenate([p['data']['data'] for p in these_projs])
            assert p.shape == (len(these_projs), len(this_evoked.data))
            traces = np.dot(p, this_evoked.data)
            traces *= np.sign(np.mean(
                np.dot(this_evoked.data, traces.T), 0))[:, np.newaxis]
            if use_ch is not None:
                ch_traces = evoked.copy().pick_channels(use_ch).data
                ch_traces -= np.mean(ch_traces, axis=1, keepdims=True)
                ch_traces /= np.abs(ch_traces).max()
            with warnings.catch_warnings(record=True):  # tight_layout
                this_evoked.plot(picks='all', axes=[ax])
            for line in ax.lines:
                line.set(lw=0.5, zorder=3)
            for t in list(ax.texts):
                t.remove()
            scale = 0.8 * np.abs(ax.get_ylim()).max()
            hs, labels = list(), list()
            traces /= np.abs(traces).max()  # uniformly scaled
            for ti, trace in enumerate(traces):
                hs.append(ax.plot(
                    this_evoked.times, trace * scale,
                    color=colors[ti % len(colors)], zorder=5)[0])
                labels.append(f'#{ti + 1}')
            traces /= np.abs(traces).max(1, keepdims=True)  # independently
            for ti, trace in enumerate(traces):
                ax.plot(
                    this_evoked.times, trace * scale,
                    color=colors[ti % len(colors)], zorder=3.5,
                    ls='--', lw=1., alpha=0.75)
            if use_ch is not None:
                hs.append(ax.plot(this_evoked.times, ch_traces.T * scale,
                                  color='#CCBB44', lw=3, zorder=4,
                                  alpha=0.75)[0])
                labels.append(kind)
            ax.set(title='', xlabel='', ylabel='')
            if need_legend and count == proj_nums.max():
                ax.legend(
                    hs, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
                    borderaxespad=0.)
                need_legend = False
            last_ax[1] = ax
            # Before and after traces
            ax = plt.subplot2grid(shape, (ri, 0), colspan=cs_trace)
            with warnings.catch_warnings(record=True):  # tight_layout
                this_evoked.plot(
                    picks='all', axes=[ax])
            for line in ax.lines:
                line.set(lw=0.5, zorder=3)
            loff = len(ax.lines)
            with warnings.catch_warnings(record=True):  # tight_layout
                this_evoked.copy().add_proj(projs).apply_proj().plot(
                    picks='all', axes=[ax])
            for line in ax.lines[loff:]:
                line.set(lw=0.5, zorder=4, color='g')
            for t in list(ax.texts):
                t.remove()
            ax.set(title='', xlabel='')
            ax.set(ylabel=f'{type_titles[ch_type]}\n{unit}')
            last_ax[0] = ax
        ri += 1
    if cs_trace:
        for ax in last_ax:
            ax.set(xlabel='Time (sec)')
    fig.subplots_adjust(0.1, 0.15, 1.0, 0.9, wspace=0.25, hspace=0.2)
    assert used.all() and (used <= 2).all()
    return fig


def _get_cov_name(p, subj, cov_name=None):
    # just the first for now
    if cov_name is None or cov_name is True:
        if p.inv_names:
            cov_name = (safe_inserter(p.inv_names[0], subj) +
                        ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
        elif p.runs_empty:  # erm cov
            new_run = safe_inserter(p.runs_empty[0], subj)
            cov_name = new_run + p.pca_extra + p.inv_tag + '-cov.fif'
    if cov_name is not None:
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        cov_name = op.join(cov_dir, safe_inserter(cov_name, subj))
        if not op.isfile(cov_name):
            cov_name = None
    return cov_name
