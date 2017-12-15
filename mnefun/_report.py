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

from ._paths import get_raw_fnames, get_report_fnames


def gen_html_report(p, subjects, structurals, run_indices=None,
                    raw=True, raw_sss=True, evoked=True, cov=True,
                    trans=True, epochs=True,
                    fwd=True, inv=True):
    """Generates HTML reports"""
    import matplotlib.pyplot as plt
    from matplotlib.image import imsave
    from ._mnefun import (_load_trans_to, plot_good_coils, _headpos,
                          _get_bem_src_trans, safe_inserter)
    if run_indices is None:
        run_indices = [None] * len(subjects)
    style = {'axes.spines.right': 'off', 'axes.spines.top': 'off',
             'axes.grid': True}
    for si, subj in enumerate(subjects):
        struc = structurals[si]
        report = Report(verbose=False)
        print('  Processing subject %s/%s (%s)'
              % (si + 1, len(subjects), subj))

        # raw
        fnames = get_raw_fnames(p, subj, 'raw', erm=False, add_splits=True,
                                run_indices=run_indices[si])
        if not all(op.isfile(fname) for fname in fnames):
            raise RuntimeError('Cannot create reports until raw data exist')
        raw = mne.concatenate_raws(
            [read_raw_fif(fname, allow_maxshield='yes')
             for fname in fnames])

        # sss
        sss_fnames = get_raw_fnames(p, subj, 'sss', False, False,
                                    run_indices[si])
        has_sss = all(op.isfile(fname) for fname in sss_fnames)
        sss_info = mne.io.read_info(sss_fnames[0]) if has_sss else None

        # pca
        pca_fnames = get_raw_fnames(p, subj, 'pca', False, False,
                                    run_indices[si])
        has_pca = all(op.isfile(fname) for fname in pca_fnames)

        with plt.style.context(style):
            ljust = 25
            #
            # Head coils
            #
            section = 'HPI coil SNR'
            if p.report_params.get('coil_snr', True) and p.movecomp:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
                if p.report_params['coil_t_step'] == 'auto':
                    t_step = raw.times[-1] / 100.  # 100 points
                else:
                    t_step = float(p.report_params['coil_t_step'])
                fig = plot_good_coils(raw, t_step, show=False)
                fig.set_size_inches(10, 2)
                fig.tight_layout()
                report.add_figs_to_section(fig, section, section,
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
                pos = [_headpos(p, fname) for ri, fname in enumerate(fnames)]
                fig = plot_head_positions(pos=pos, destination=trans_to,
                                          info=raw.info, show=False)
                del trans_to
                fig.set_size_inches(10, 6)
                fig.tight_layout()
                report.add_figs_to_section(fig, section, section,
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
                captions = []
                figs = []
                if p.proj_extra is not None:
                    captions.append('%s: Custom' % section)
                    projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                              p.proj_extra))
                    figs.append(plot_projs_topomap(projs, info=sss_info,
                                                   show=False))
                if any(p.proj_nums[0]):  # ECG
                    captions.append('%s: ECG' % section)
                    projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                              'preproc_ecg-proj.fif'))
                    figs.append(plot_projs_topomap(projs, info=sss_info,
                                                   show=False))
                if any(p.proj_nums[1]):  # EOG
                    captions.append('%s: Blink' % section)
                    projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                              'preproc_blink-proj.fif'))
                    figs.append(plot_projs_topomap(projs, info=sss_info,
                                                   show=False))
                if any(p.proj_nums[2]):  # ERM
                    captions.append('%s: Continuous' % section)
                    projs = read_proj(op.join(p.work_dir, subj, p.pca_dir,
                                              'preproc_cont-proj.fif'))
                    figs.append(plot_projs_topomap(projs, info=sss_info,
                                                   show=False))
                report.add_figs_to_section(figs, captions, section,
                                           image_format='svg')
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
                captions = ['Left', 'Front', 'Right']
                captions = ['%s: %s' % (section, c) for c in captions]
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
                    tempdir = mne.utils._TempDir()
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
                        images = list()
                        for ai, angle in enumerate([180, 90, 0]):
                            mlab.view(angle, 90, focalpoint=(0., 0., 0.),
                                      distance=0.6, figure=fig)
                            view = mlab.screenshot(figure=fig)
                            view = view[:, (view != 0).any(0).any(-1)]
                            view = view[(view != 0).any(1).any(-1)]
                            images.append(op.join(tempdir, '%s.png' % ai))
                            imsave(images[-1], view)
                        mlab.close(fig)
                        report.add_images_to_section(
                            images, captions=captions, section=section)
                    finally:
                        del tempdir
                        mlab.options.offscreen = offscreen
                print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)
            #
            # BEM
            #
            section = 'BEM'
            if p.report_params.get('bem', True):
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
                        print(('    %s ... ' % section).ljust(ljust), end='')
                        report.add_bem_to_section(struc, caption, section,
                                                  decim=10, n_jobs=1)
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

                whitening = p.report_params['whitening']
                assert isinstance(whitening, dict)
                analysis = whitening['analysis']
                name = whitening['name']
                cov = whitening['cov']
                # Load the inverse
                cov_dir = op.join(p.work_dir, subj, p.cov_dir)
                cov_name = op.join(cov_dir, safe_inserter(cov, subj))
                inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
                fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                       % (analysis, p.lp_cut, p.inv_tag,
                                          p.eq_tag, subj))
                if not op.isfile(cov_name):
                    print('Missing cov: %s' % cov_name)
                elif not op.isfile(fname_evoked):
                    print('Missing evoked: %s' % fname_evoked)
                else:
                    cov = mne.read_cov(cov_name)
                    evo = mne.read_evokeds(fname_evoked, name)
                    comments = '%s["%s"] (N=%d)' % (analysis, name, evo.nave)
                    fig = evo.plot_white(cov)
                    report.add_figs_to_section(
                        fig, captions=section, comments=comments,
                        section=section, image_format='png')
                    print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

            #
            # Source estimation
            #
            section = 'Source estimation'
            if p.report_params.get('source', None) is not None:
                t0 = time.time()
                print(('    %s ... ' % section).ljust(ljust), end='')
                source = p.report_params['source']
                assert isinstance(source, dict)
                analysis = source['analysis']
                name = source['name']
                # Load the inverse
                inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
                fname_inv = op.join(inv_dir,
                                    safe_inserter(source['inv'], subj))
                inv = mne.minimum_norm.read_inverse_operator(fname_inv)
                fname_evoked = op.join(inv_dir, '%s_%d%s_%s_%s-ave.fif'
                                       % (analysis, p.lp_cut, p.inv_tag,
                                          p.eq_tag, subj))
                if not op.isfile(fname_inv):
                    print('Missing inv: %s' % fname_inv)
                elif not op.isfile(fname_evoked):
                    print('Missing evoked: %s' % fname_evoked)
                else:
                    evoked = mne.read_evokeds(fname_evoked, name)
                    title = '%s["%s"] (N=%d)' % (analysis, name, evoked.nave)
                    stc = mne.minimum_norm.apply_inverse(
                        evoked, inv, lambda2=source.get('lambda2', 1. / 9.),
                        method=source.get('method', 'dSPM'),
                        pick_ori=None)
                    stc = abs(stc)
                    # get clim using the reject_tmin <->reject_tmax
                    stc_crop = stc.copy().crop(p.reject_tmin, p.reject_tmax)
                    clim = source.get('clim', dict(kind='percent',
                                                   lims=[82, 90, 98]))
                    clim = mne.viz._3d._limits_to_control_points(
                         clim, stc_crop.data, 'viridis')[0]  # cmap is dummy
                    clim = dict(kind='value', lims=clim)
                    if not isinstance(stc, mne.SourceEstimate):
                        print('Only surface source estimates currently '
                              'supported')
                    else:
                        brain = stc.plot(
                            hemi=source.get('hemi', 'split'),
                            views=source.get('views', ['lat', 'med']),
                            size=source.get('size', (800, 600)),
                            colormap=source.get('colormap', 'viridis'),
                            transparent=source.get('transparent', True),
                            foreground='k', background='w',
                            clim=clim,
                            )
                        imgs = list()
                        times = source.get('times', [0.1])
                        for t in times:
                            brain.set_time(t)
                            imgs.append(brain.screenshot())
                        brain.close()
                        captions = ['%2.3f sec' % t for t in times]
                        report.add_slider_to_section(
                            imgs, captions=times, section=section,
                            title=title, image_format='png')
                        print('%5.1f sec' % ((time.time() - t0),))
            else:
                print('    %s skipped' % section)

        report_fname = get_report_fnames(p, subj)[0]
        report.save(report_fname, open_browser=False, overwrite=True)
