#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Create HTML reports.
"""
from __future__ import print_function, unicode_literals

import os.path as op
import time
import warnings

import numpy as np

import mne
from mne import read_proj
from mne.io import read_raw_fif
from mne.viz import plot_projs_topomap
from mne.viz._3d import plot_head_positions
from mne.viz import plot_snr_estimate
from mne.report import Report

from ._paths import get_raw_fnames, get_proj_fnames, get_report_fnames


def gen_html_report(p, subjects, structurals, run_indices=None):
    """Generates HTML reports"""
    import matplotlib.pyplot as plt
    from ._mnefun import (_load_trans_to, plot_good_coils, _head_pos_annot,
                          _get_bem_src_trans, safe_inserter, _prebad,
                          _load_meg_bads, mlab_offscreen, _fix_raw_eog_cals)
    if run_indices is None:
        run_indices = [None] * len(subjects)
    style = {'axes.spines.right': 'off', 'axes.spines.top': 'off',
             'axes.grid': True}
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
        raw = [read_raw_fif(fname, allow_maxshield='yes')
               for fname in fnames]
        _fix_raw_eog_cals(raw)
        raw = mne.concatenate_raws(raw)
        prebad_file = _prebad(p, subj)
        _load_meg_bads(raw, prebad_file, disp=False)

        # sss
        sss_fnames = get_raw_fnames(p, subj, 'sss', False, False,
                                    run_indices[si])
        has_sss = all(op.isfile(fname) for fname in sss_fnames)
        sss_info = mne.io.read_info(sss_fnames[0]) if has_sss else None

        # pca
        pca_fnames = get_raw_fnames(p, subj, 'pca', False, False,
                                    run_indices[si])
        has_pca = all(op.isfile(fname) for fname in pca_fnames)

        # whitening and source localization
        inv_dir = op.join(p.work_dir, subj, p.inverse_dir)

        has_fwd = op.isfile(op.join(p.work_dir, subj, p.forward_dir,
                                    subj + p.inv_tag + '-fwd.fif'))

        with plt.style.context(style):
            ljust = 25
            #
            # Head coils
            #
            section = 'Good HPI count'
            if p.report_params.get('good_hpi_count', True) and p.movecomp:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
                figs = list()
                captions = list()
                for fname in fnames:
                    _, _, fit_data = _head_pos_annot(p, fname, prefix='      ')
                    fig = plot_good_coils(fit_data, show=False)
                    fig.set_size_inches(10, 2)
                    fig.tight_layout()
                    figs.append(fig)
                    captions.append('%s: %s' % (section, op.split(fname)[-1]))
                report.add_figs_to_section(figs, captions, section,
                                           image_format='svg')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # Head movement
            #
            section = 'Head movement'
            if p.report_params.get('head_movement', True) and p.movecomp:
                print(('    %s ... ' % section).ljust(ljust), end='')
                t0 = time.time()
                trans_to = _load_trans_to(p, subj, run_indices[si], raw)
                figs = list()
                captions = list()
                for fname in fnames:
                    pos, _, _ = _head_pos_annot(p, fname, prefix='      ')
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
            if op.isfile(pca_fnames[0]):
                raw_pca = mne.concatenate_raws(
                    [mne.io.read_raw_fif(fname) for fname in pca_fnames])
            section = 'Raw segments'
            if p.report_params.get('raw_segments', True) and has_pca:
                events = np.linspace(raw.times[0], raw.times[-1], 12)[1:-1]
                events = np.round(events * raw.info['sfreq']).astype(int)
                events += raw.first_samp
                events = np.array([events,
                                   np.zeros_like(events),
                                   np.ones_like(events)]).T
                epochs = mne.Epochs(raw_pca, events, tmin=-0.5, tmax=0.5,
                                    baseline=(None, None), preload=True,
                                    reject_by_annotation=False)
                event_times = ((epochs.events[:, 0] - raw.first_samp) /
                               raw.info['sfreq'])
                event_times = ['%0.1f' % e for e in event_times]
                raw_plot = mne.io.RawArray(
                    epochs.get_data()
                    .transpose(1, 0, 2).reshape(len(epochs.ch_names), -1),
                    epochs.info)
                new_events = np.linspace(
                    0, int(round(10 * raw.info['sfreq'])) - 1, 11).astype(int)
                new_events += raw_plot.first_samp
                new_events = np.array([new_events,
                                       np.zeros_like(new_events),
                                       np.ones_like(new_events)]).T
                fig = raw_plot.plot(group_by='selection', butterfly=True,
                                    events=new_events)
                fig.axes[0].lines[-1].set_zorder(10)  # events
                fig.axes[0].set(xticks=np.arange(0, len(event_times)) + 0.5)
                fig.axes[0].set(xticklabels=event_times)
                fig.axes[0].set(xlabel='Center of 1-second segments')
                fig.axes[0].grid(False)
                for _ in range(len(fig.axes) - 1):
                    fig.delaxes(fig.axes[-1])
                fig.set(figheight=(fig.axes[0].get_yticks() != 0).sum(),
                        figwidth=12)
                fig.subplots_adjust(0.0, 0.0, 1, 1, 0, 0)
                report.add_figs_to_section(fig, 'Processed', section,
                                           image_format='png')  # svd too slow

            #
            # PSD
            #
            section = 'PSD'
            if p.report_params.get('psd', True) and has_pca:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
                if p.lp_trans == 'auto':
                    lp_trans = 0.25 * p.lp_cut
                else:
                    lp_trans = p.lp_trans
                n_fft = 8192
                fmax = raw.info['lowpass']
                figs = [raw.plot_psd(fmax=fmax, n_fft=n_fft, show=False)]
                captions = ['%s: Raw' % section]
                fmax = p.lp_cut + 2 * lp_trans
                figs.append(raw.plot_psd(fmax=fmax, n_fft=n_fft, show=False))
                captions.append('%s: Raw (zoomed)' % section)
                if op.isfile(pca_fnames[0]):
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
            else:
                print('    %s skipped' % section)

            #
            # SSP
            #
            section = 'SSP topomaps'

            if p.report_params.get('ssp_topomaps', True) and has_pca and \
                    np.sum(_get_proj_nums(p, subj)) > 0:
                assert sss_info is not None
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
                figs = []
                comments = []
                proj_files = get_proj_fnames(p, subj)
                if p.proj_extra is not None:
                    comments.append('Custom')
                    projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                              p.proj_extra))
                    figs.append(plot_projs_topomap(projs, info=sss_info,
                                                   show=False))
                if any(p.proj_nums[0]):  # ECG
                    if 'preproc_ecg-proj.fif' in proj_files:
                        comments.append('ECG')
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_ecg-proj.fif'), sss_info))
                if any(p.proj_nums[1]):  # EOG
                    if 'preproc_blink-proj.fif' in proj_files:
                        comments.append('Blink')
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_blink-proj.fif'), sss_info))
                if any(p.proj_nums[2]):  # ERM
                    if 'preproc_blink-cont.fif' in proj_files:
                        comments.append('Continuous')
                        figs.append(_proj_fig(op.join(
                            p.work_dir, subj, p.pca_dir,
                            'preproc_cont-proj.fif'), sss_info))
                # adjust sizes
                for fig in figs:
                    n_rows = np.floor(np.sqrt(len(fig.axes)))
                    n_cols = np.ceil(len(fig.axes) / float(n_rows))
                    fig.set_size_inches(2 * n_cols, 2 * n_rows)
                    fig.tight_layout()
                captions = [section] + [None] * (len(comments) - 1)
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
            if p.report_params.get('source_alignment', True) and has_sss \
                    and has_fwd:
                assert sss_info is not None
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
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
                        kwargs = dict(
                            info=sss_info, subjects_dir=subjects_dir, bem=bem,
                            dig=True, coord_frame=coord_frame, show_axes=True,
                            fig=fig, trans=trans, src=src)
                        try_surfs = ['head-dense', 'head', 'inner_skull']
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
                    view = np.concatenate(view, axis=1)
                    view = view[:, (view != 0).any(0).any(-1)]
                    view = view[(view != 0).any(1).any(-1)]
                    report.add_figs_to_section(view, captions, section)
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)
            #
            # SNR
            #
            section = 'SNR'
            if p.report_params.get('snr', None) is not None:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
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
                        title = ('%s<br>%s["%s"] (N=%d)'
                                 % (section, analysis, name, this_evoked.nave))
                        figs = plot_snr_estimate(this_evoked, inv,
                                                 verbose=False)
                        figs.axes[0].set_ylim(auto=True)
                        captions = ('%s<br>%s["%s"] (N=%d)'
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
                caption = '%s<br>%s' % (section, struc)
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
                        print(('    %s ... ' % section).ljust(ljust), end='')
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
            section = 'Whitening'
            if p.report_params.get('whitening', False):
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')

                whitenings = p.report_params['whitening']
                if not isinstance(whitenings, (list, tuple)):
                    whitenings = [whitenings]
                for whitening in whitenings:
                    assert isinstance(whitening, dict)
                    analysis = whitening['analysis']
                    name = whitening['name']
                    cov_name = op.join(p.work_dir, subj, p.cov_dir,
                                       safe_inserter(whitening['cov'], subj))
                    # Load the inverse
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if not op.isfile(cov_name):
                        print('    Missing cov: %s'
                              % op.basename(cov_name), end='')
                    elif not op.isfile(fname_evoked):
                        print('    Missing evoked: %s'
                              % op.basename(fname_evoked), end='')
                    else:
                        noise_cov = mne.read_cov(cov_name)
                        evo = mne.read_evokeds(fname_evoked, name)
                        captions = ('%s<br>%s["%s"] (N=%d)'
                                    % (section, analysis, name, evo.nave))
                        fig = evo.plot_white(noise_cov, **time_kwargs)
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
                print(('    %s ... ' % section).ljust(ljust), end='')
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
                        captions = ('%s<br>%s["%s"] (N=%d)'
                                    % (section, analysis, name,
                                       this_evoked.nave))
                        captions = [captions] + [None] * (len(figs) - 1)
                        report.add_figs_to_section(
                            figs, captions, section=section,
                            image_format='svg')
                print('%5.1f sec' % ((time.time() - t0),))

            #
            # Source estimation
            #
            section = 'Source estimation'
            if p.report_params.get('source', False):
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
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
                        title = ('%s<br>%s["%s"] (N=%d)'
                                 % (section, analysis, name, this_evoked.nave))
                        stc = mne.minimum_norm.apply_inverse(
                            this_evoked, inv,
                            lambda2=source.get('lambda2', 1. / 9.),
                            method=source.get('method', 'dSPM'),
                            pick_ori=None)
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
                        if not isinstance(stc, mne.SourceEstimate):
                            print('Only surface source estimates currently '
                                  'supported')
                        else:
                            subjects_dir = mne.utils.get_subjects_dir(
                                p.subjects_dir, raise_error=True)
                            with mlab_offscreen():
                                brain = stc.plot(
                                    hemi=source.get('hemi', 'split'),
                                    views=source.get('views', ['lat', 'med']),
                                    size=source.get('size', (800, 600)),
                                    colormap=source.get('colormap', 'viridis'),
                                    transparent=source.get('transparent',
                                                           True),
                                    foreground='k', background='w',
                                    clim=clim, subjects_dir=subjects_dir,
                                    )
                                imgs = list()
                                for t in times:
                                    brain.set_time(t)
                                    imgs.append(brain.screenshot())
                                brain.close()
                            captions = ['%2.3f sec' % t for t in times]
                            report.add_slider_to_section(
                                imgs, captions=captions, section=section,
                                title=title, image_format='png')
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

        report_fname = get_report_fnames(p, subj)[0]
        report.save(report_fname, open_browser=False, overwrite=True)


def _proj_fig(fname, info):
    import matplotlib.pyplot as plt
    projs = read_proj(fname)
    epochs = fname.replace('-proj.fif', '-epo.fif')
    if op.isfile(epochs):
        epochs = mne.read_epochs(epochs)
        evoked = epochs.average()
        n_row = 2
    else:
        n_row = 1
    fig, axes = plt.subplots(n_row, len(projs),
                             figsize=(len(projs) * 1.5, n_row * 2))
    plot_projs_topomap(projs, info=info, show=False, axes=axes[0])
    for pi, proj in enumerate(projs):
        axes[0, pi].set(title='%s %s' % (proj['desc'].split('-')[0],
                                         proj['desc'].split('-')[-1]))
    if n_row == 2:
        for pi, proj in enumerate(projs):
            ax = axes[1, pi]
            p = proj['data']['data']
            this_evoked = evoked.copy().pick_channels(
                proj['data']['col_names'])
            assert p.shape == (1, len(this_evoked.data))
            with warnings.catch_warnings(record=True):  # tight_layout
                this_evoked.plot(
                    picks=np.arange(len(this_evoked.data)), axes=[ax])
            ax.texts = []
            trace = np.dot(p, this_evoked.data)[0]
            trace *= 0.8 * np.abs(ax.get_ylim()).max() / np.abs(trace).max()
            ax.plot(this_evoked.times, trace, color='#9467bd')
            ax.set(title='', ylabel='')
        axes[1, 0].set(ylabel='N$_{ave}$=%d' % (evoked.nave,))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig
