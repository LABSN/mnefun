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
from mne.report import Report

from ._paths import get_raw_fnames, get_proj_fnames, get_report_fnames


def gen_html_report(p, subjects, structurals, run_indices=None):
    """Generates HTML reports"""
    import matplotlib.pyplot as plt
    from ._mnefun import (_load_trans_to, plot_good_coils, _head_pos_annot,
                          _get_bem_src_trans, safe_inserter, _prebad,
                          _load_meg_bads)
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
        raw = mne.concatenate_raws(
            [read_raw_fif(fname, allow_maxshield='yes')
             for fname in fnames])
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
                    raw_pca = mne.concatenate_raws(
                        [mne.io.read_raw_fif(fname) for fname in pca_fnames])
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
            if p.report_params.get('ssp_topomaps', True) and has_pca:
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
                        projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                                  'preproc_ecg-proj.fif'))
                        figs.append(plot_projs_topomap(projs, info=sss_info,
                                                       show=False))
                if any(p.proj_nums[1]):  # EOG
                    if 'preproc_blink-proj.fif' in proj_files:
                        comments.append('Blink')
                        projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                                  'preproc_blink-proj.fif'))
                        figs.append(plot_projs_topomap(projs, info=sss_info,
                                                       show=False))
                if any(p.proj_nums[2]):  # ERM
                    if 'preproc_blink-cont.fif' in proj_files:
                        comments.append('Continuous')
                        projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                                  'preproc_cont-proj.fif'))
                        figs.append(plot_projs_topomap(projs, info=sss_info,
                                                       show=False))
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
            if p.report_params.get('source_alignment', True) and has_sss:
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
                    offscreen = mlab.options.offscreen
                    mlab.options.offscreen = True
                    if len(mne.pick_types(sss_info)):
                        coord_frame = 'meg'
                    else:
                        coord_frame = 'head'
                    try:
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
                    finally:
                        mlab.options.offscreen = offscreen
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)
            #
            # BEM
            #
            section = 'BEM'
            if p.report_params.get('bem', True):
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
            if p.report_params.get('whitening', None) is not None:
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
                        print('Missing cov: %s' % cov_name)
                    elif not op.isfile(fname_evoked):
                        print('Missing evoked: %s' % fname_evoked)
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
            if p.report_params.get('sensor', None) is not None:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
                sensors = p.report_params['sensor']
                if not isinstance(sensors, (list, tuple)):
                    sensors = [sensors]
                for sensor in sensors:
                    assert isinstance(sensor, dict)
                    analysis = sensor['analysis']
                    name = sensor['name']
                    times = sensor.get('times', [0.1])
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if not op.isfile(fname_evoked):
                        print('Missing evoked: %s' % fname_evoked)
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
            if p.report_params.get('source', None) is not None:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
                sources = p.report_params['source']
                if not isinstance(sources, (list, tuple)):
                    sources = [sources]
                for source in sources:
                    assert isinstance(source, dict)
                    analysis = source['analysis']
                    name = source['name']
                    times = source.get('times', [0.1])
                    # Load the inverse
                    inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
                    fname_inv = op.join(inv_dir,
                                        safe_inserter(source['inv'], subj))
                    fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                           % (analysis, p.lp_cut, p.inv_tag,
                                              p.eq_tag, subj))
                    if not op.isfile(fname_inv):
                        print('Missing inv: %s' % fname_inv)
                    elif not op.isfile(fname_evoked):
                        print('Missing evoked: %s' % fname_evoked)
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
                        clim = mne.viz._3d._limits_to_control_points(
                             clim, stc_crop.data, 'viridis')[0]  # dummy cmap
                        clim = dict(kind='value', lims=clim)
                        if not isinstance(stc, mne.SourceEstimate):
                            print('Only surface source estimates currently '
                                  'supported')
                        else:
                            subjects_dir = mne.utils.get_subjects_dir(
                                p.subjects_dir, raise_error=True)
                            brain = stc.plot(
                                hemi=source.get('hemi', 'split'),
                                views=source.get('views', ['lat', 'med']),
                                size=source.get('size', (800, 600)),
                                colormap=source.get('colormap', 'viridis'),
                                transparent=source.get('transparent', True),
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
