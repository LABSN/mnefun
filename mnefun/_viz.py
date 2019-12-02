"""Visualization helpers."""

from contextlib import contextmanager
import copy

import os.path as op
import numpy as np
from scipy import linalg

from mne import read_proj, read_events, pick_types
from mne.utils import verbose
from mne.viz.utils import tight_layout, plt_show

from ._sss import compute_good_coils
from ._paths import get_raw_fnames


def _viz_raw_ssp_events(p, subj, ridx):
    """Plot filtered cleaned raw trace with ExG events"""
    from ._ssp import _raw_LRFCP
    pca_dir = op.join(p.work_dir, subj, p.pca_dir)
    raw_names = get_raw_fnames(p, subj, 'sss', False, False, ridx)
    pre_list = [r for ri, r in enumerate(raw_names)
                if ri in p.get_projs_from]
    all_proj = op.join(pca_dir, 'preproc_all-proj.fif')
    projs = read_proj(all_proj)
    colors = dict()
    ev = np.zeros((0, 3), int)
    for n, c, cid in zip(['ecg', 'blink'], ['r', 'b'], [999, 998]):
        fname = op.join(pca_dir, 'preproc_%s-eve.fif' % n)
        if op.isfile(fname):
            ev = np.concatenate((ev, read_events(fname)))
            colors[cid] = c
    ev = ev[np.argsort(ev[:, 0], axis=0)]
    raw = _raw_LRFCP(pre_list, p.proj_sfreq, None, None, p.n_jobs_fir,
                     p.n_jobs_resample, projs, None, p.disp_files,
                     method='fir', filter_length=p.filter_length,
                     force_bads=False, l_trans=p.hp_trans, h_trans=p.lp_trans)
    raw.plot(events=ev, event_color=colors)


def clean_brain(brain_img):
    """Remove borders of a brain image and make transparent."""
    bg = (brain_img == brain_img[0, 0]).all(-1)
    brain_img = brain_img[(~bg).any(axis=-1)]
    brain_img = brain_img[:, (~bg).any(axis=0)]
    alpha = 255 * np.ones(brain_img.shape[:-1], np.uint8)
    x, y = np.where((brain_img == 255).all(-1))
    alpha[x, y] = 0
    return np.concatenate((brain_img, alpha[..., np.newaxis]), -1)


def plot_colorbar(lims, ticks=None, ticklabels=None, figsize=(1, 2),
                  labelsize='small', ticklabelsize='x-small', ax=None,
                  label='', tickrotation=0., orientation='vertical',
                  end_labels=None, colormap='mne', transparent=False,
                  diverging=None):
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    from mne.viz._3d import _limits_to_control_points
    with plt.rc_context({'axes.labelsize': labelsize,
                         'xtick.labelsize': ticklabelsize,
                         'ytick.labelsize': ticklabelsize}):
        if diverging is None:
            diverging = (colormap == 'mne')  # simple heuristic here
        if diverging:
            use_lims = dict(kind='value', pos_lims=lims)
        else:
            use_lims = dict(kind='value', lims=lims)
        cmap, scale_pts, diverging, _, none_ticks = _limits_to_control_points(
            use_lims, 0, colormap, transparent, linearize=True)
        vmin, vmax = scale_pts[0], scale_pts[-1]
        if ticks is None:
            ticks = none_ticks
        del colormap, lims, use_lims
        adjust = (ax is None)
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            fig = ax.figure
        norm = Normalize(vmin=vmin, vmax=vmax)
        if ticklabels is None:
            ticklabels = ticks
        assert len(ticks) == len(ticklabels)
        cbar = ColorbarBase(ax, cmap, norm=norm, ticks=ticks, label=label,
                            orientation=orientation)
        for key in ('left', 'top',
                    'bottom' if orientation == 'vertical' else 'right'):
            ax.spines[key].set_visible(False)
        cbar.set_ticklabels(ticklabels)
        cbar.patch.set(facecolor='0.5', edgecolor='0.5')
        if orientation == 'horizontal':
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=tickrotation)
        else:
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=tickrotation)
        cbar.outline.set_visible(False)
        lims = np.array(list(ax.get_xlim()) + list(ax.get_ylim()))
        if end_labels is not None:
            if orientation == 'horizontal':
                delta = np.diff(lims[:2]) * np.array([-0.05, 0.05])
                xs = np.array(lims[:2]) + delta
                has = ['right', 'left']
                ys = [lims[2:].mean()] * 2
                vas = ['center', 'center']
            else:
                xs = [lims[:2].mean()] * 2
                has = ['center'] * 2
                delta = np.diff(lims[2:]) * np.array([-0.05, 0.05])
                ys = lims[2:] + delta
                vas = ['top', 'bottom']
            for x, y, l, ha, va in zip(xs, ys, end_labels, has, vas):
                ax.text(x, y, l, ha=ha, va=va, fontsize=ticklabelsize)
        if adjust:
            fig.subplots_adjust(0.01, 0.05, 0.2, 0.95)
    return fig


def plot_reconstruction(evoked, origin=(0., 0., 0.04)):
    """Plot the reconstructed data for Evoked

    Currently only works for MEG data.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data.
    origin : array-like, shape (3,)
        The head origin to use.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    from mne.forward._field_interpolation import _map_meg_channels
    import matplotlib.pyplot as plt
    evoked = evoked.copy().pick_types(meg=True, exclude='bads')
    info_to = copy.deepcopy(evoked.info)
    info_to['projs'] = []
    op = _map_meg_channels(
        evoked.info, info_to, mode='accurate', origin=(0., 0., 0.04))
    fig, axs = plt.subplots(3, 2, squeeze=False)
    titles = dict(grad='Gradiometers (fT/cm)', mag='Magnetometers (fT)')
    for mi, meg in enumerate(('grad', 'mag')):
        picks = pick_types(evoked.info, meg=meg)
        kwargs = dict(ylim=dict(grad=[-250, 250], mag=[-600, 600]),
                      spatial_colors=True, picks=picks)
        evoked.plot(axes=axs[0, mi], proj=False,
                    titles=dict(grad='Proj off', mag=''), **kwargs)
        evoked_remap = evoked.copy().apply_proj()
        evoked_remap.info['projs'] = []
        evoked_remap.plot(axes=axs[1, mi],
                          titles=dict(grad='Proj on', mag=''), **kwargs)
        evoked_remap.data = np.dot(op, evoked_remap.data)
        evoked_remap.plot(axes=axs[2, mi],
                          titles=dict(grad='Recon', mag=''), **kwargs)
        axs[0, mi].set_title(titles[meg])
        for ii in range(3):
            if ii in (0, 1):
                axs[ii, mi].set_xlabel('')
            if ii in (1, 2):
                axs[ii, mi].set_title('')
    for ii in range(3):
        axs[ii, 1].set_ylabel('')
    axs[0, 0].set_ylabel('Original')
    axs[1, 0].set_ylabel('Projection')
    axs[2, 0].set_ylabel('Reconstruction')
    fig.tight_layout()
    return fig


def plot_chpi_snr_raw(raw, win_length, n_harmonics=None, show=True,
                      verbose=True):
    """Compute and plot cHPI SNR from raw data

    Parameters
    ----------
    win_length : float
        Length of window to use for SNR estimates (seconds). A longer window
        will naturally include more low frequency power, resulting in lower
        SNR.
    n_harmonics : int or None
        Number of line frequency harmonics to include in the model. If None,
        use all harmonics up to the MEG analog lowpass corner.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        cHPI SNR as function of time, residual variance.

    Notes
    -----
    A general linear model including cHPI and line frequencies is fit into
    each data window. The cHPI power obtained from the model is then divided
    by the residual variance (variance of signal unexplained by the model) to
    obtain the SNR.

    The SNR may decrease either due to decrease of cHPI amplitudes (e.g.
    head moving away from the helmet), or due to increase in the residual
    variance. In case of broadband interference that overlaps with the cHPI
    frequencies, the resulting decreased SNR accurately reflects the true
    situation. However, increased narrowband interference outside the cHPI
    and line frequencies would also cause an increase in the residual variance,
    even though it wouldn't necessarily affect estimation of the cHPI
    amplitudes. Thus, this method is intended for a rough overview of cHPI
    signal quality. A more accurate picture of cHPI quality (at an increased
    computational cost) can be obtained by examining the goodness-of-fit of
    the cHPI coil fits.
    """
    import matplotlib.pyplot as plt
    from mne.chpi import _get_hpi_info

    # plotting parameters
    legend_fontsize = 10
    title_fontsize = 10
    tick_fontsize = 10
    label_fontsize = 10

    # get some info from fiff
    sfreq = raw.info['sfreq']
    linefreq = raw.info['line_freq']
    if n_harmonics is not None:
        linefreqs = (np.arange(n_harmonics + 1) + 1) * linefreq
    else:
        linefreqs = np.arange(linefreq, raw.info['lowpass'], linefreq)
    buflen = int(win_length * sfreq)
    if buflen <= 0:
        raise ValueError('Window length should be >0')
    cfreqs = _get_hpi_info(raw.info)[0]
    if verbose:
        print('Nominal cHPI frequencies: %s Hz' % cfreqs)
        print('Sampling frequency: %s Hz' % sfreq)
        print('Using line freqs: %s Hz' % linefreqs)
        print('Using buffers of %s samples = %s seconds\n'
              % (buflen, buflen / sfreq))

    pick_meg = pick_types(raw.info, meg=True, exclude=[])
    pick_mag = pick_types(raw.info, meg='mag', exclude=[])
    pick_grad = pick_types(raw.info, meg='grad', exclude=[])
    nchan = len(pick_meg)
    # grad and mag indices into an array that already has meg channels only
    pick_mag_ = np.in1d(pick_meg, pick_mag).nonzero()[0]
    pick_grad_ = np.in1d(pick_meg, pick_grad).nonzero()[0]

    # create general linear model for the data
    t = np.arange(buflen) / float(sfreq)
    model = np.empty((len(t), 2 + 2 * (len(linefreqs) + len(cfreqs))))
    model[:, 0] = t
    model[:, 1] = np.ones(t.shape)
    # add sine and cosine term for each freq
    allfreqs = np.concatenate([linefreqs, cfreqs])
    model[:, 2::2] = np.cos(2 * np.pi * t[:, np.newaxis] * allfreqs)
    model[:, 3::2] = np.sin(2 * np.pi * t[:, np.newaxis] * allfreqs)
    inv_model = linalg.pinv(model)

    # drop last buffer to avoid overrun
    bufs = np.arange(0, raw.n_times, buflen)[:-1]
    tvec = bufs / sfreq
    snr_avg_grad = np.zeros([len(cfreqs), len(bufs)])
    hpi_pow_grad = np.zeros([len(cfreqs), len(bufs)])
    snr_avg_mag = np.zeros([len(cfreqs), len(bufs)])
    resid_vars = np.zeros([nchan, len(bufs)])
    for ind, buf0 in enumerate(bufs):
        if verbose:
            print('Buffer %s/%s' % (ind + 1, len(bufs)))
        megbuf = raw[pick_meg, buf0:buf0 + buflen][0].T
        coeffs = np.dot(inv_model, megbuf)
        coeffs_hpi = coeffs[2 + 2 * len(linefreqs):]
        resid_vars[:, ind] = np.var(megbuf - np.dot(model, coeffs), 0)
        # get total power by combining sine and cosine terms
        # sinusoidal of amplitude A has power of A**2/2
        hpi_pow = (coeffs_hpi[0::2, :] ** 2 + coeffs_hpi[1::2, :] ** 2) / 2
        hpi_pow_grad[:, ind] = hpi_pow[:, pick_grad_].mean(1)
        # divide average HPI power by average variance
        snr_avg_grad[:, ind] = hpi_pow_grad[:, ind] / \
            resid_vars[pick_grad_, ind].mean()
        snr_avg_mag[:, ind] = hpi_pow[:, pick_mag_].mean(1) / \
            resid_vars[pick_mag_, ind].mean()

    cfreqs_legend = ['%s Hz' % fre for fre in cfreqs]
    fig, axs = plt.subplots(4, 1, sharex=True)

    # SNR plots for gradiometers and magnetometers
    ax = axs[0]
    lines1 = ax.plot(tvec, 10 * np.log10(snr_avg_grad.T))
    lines1_med = ax.plot(tvec, 10 * np.log10(np.median(snr_avg_grad, axis=0)),
                         lw=2, ls=':', color='k')
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='SNR (dB)')
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Mean cHPI power / mean residual variance, gradiometers',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax = axs[1]
    lines2 = ax.plot(tvec, 10 * np.log10(snr_avg_mag.T))
    lines2_med = ax.plot(tvec, 10 * np.log10(np.median(snr_avg_mag, axis=0)),
                         lw=2, ls=':', color='k')
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='SNR (dB)')
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Mean cHPI power / mean residual variance, magnetometers',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax = axs[2]
    lines3 = ax.plot(tvec, hpi_pow_grad.T)
    lines3_med = ax.plot(tvec, np.median(hpi_pow_grad, axis=0),
                         lw=2, ls=':', color='k')
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='Power (T/m)$^2$')
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Mean cHPI power, gradiometers',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # residual (unexplained) variance as function of time
    ax = axs[3]
    cls = plt.get_cmap('plasma')(np.linspace(0., 0.7, len(pick_meg)))
    ax.set_prop_cycle(color=cls)
    ax.semilogy(tvec, resid_vars[pick_grad_, :].T, alpha=.4)
    ax.set_xlim([tvec.min(), tvec.max()])
    ax.set(ylabel='Var. (T/m)$^2$', xlabel='Time (s)')
    ax.xaxis.label.set_fontsize(label_fontsize)
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.set_title('Residual (unexplained) variance, all gradiometer channels',
                 fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    tight_layout(pad=.5, w_pad=.1, h_pad=.2)  # from mne.viz
    # tight_layout will screw these up
    ax = axs[0]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # order curve legends according to mean of data
    sind = np.argsort(snr_avg_grad.mean(axis=1))[::-1]
    handles = [lines1[i] for i in sind]
    handles.append(lines1_med[0])
    labels = [cfreqs_legend[i] for i in sind]
    labels.append('Median')
    ax.legend(handles, labels,
              prop={'size': legend_fontsize}, bbox_to_anchor=(1.02, 0.5, ),
              loc='center left', borderpad=1)
    ax = axs[1]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    sind = np.argsort(snr_avg_mag.mean(axis=1))[::-1]
    handles = [lines2[i] for i in sind]
    handles.append(lines2_med[0])
    labels = [cfreqs_legend[i] for i in sind]
    labels.append('Median')
    ax.legend(handles, labels,
              prop={'size': legend_fontsize}, bbox_to_anchor=(1.02, 0.5, ),
              loc='center left', borderpad=1)
    ax = axs[2]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    sind = np.argsort(hpi_pow_grad.mean(axis=1))[::-1]
    handles = [lines3[i] for i in sind]
    handles.append(lines3_med[0])
    labels = [cfreqs_legend[i] for i in sind]
    labels.append('Median')
    ax.legend(handles, labels,
              prop={'size': legend_fontsize}, bbox_to_anchor=(1.02, 0.5, ),
              loc='center left', borderpad=1)
    ax = axs[3]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if show:
        plt.show()

    return fig


@verbose
def plot_good_coils(raw, t_step=1., t_window=0.2, dist_limit=0.005,
                    show=True, verbose=None):
    """Plot the good coil count as a function of time."""
    import matplotlib.pyplot as plt
    if isinstance(raw, dict):  # fit_data calculated and stored to disk
        t = raw['fit_t']
        counts = raw['counts']
        n_coils = raw['n_coils']
    else:
        t, counts, n_coils = compute_good_coils(raw, t_step, t_window,
                                                dist_limit)
    del t_step, t_window, dist_limit
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.step(t, counts, zorder=4, color='k', clip_on=False)
    ax.set(xlim=t[[0, -1]], ylim=[0, n_coils], xlabel='Time (sec)',
           ylabel='Good coils')
    ax.set(yticks=np.arange(n_coils + 1))
    for comp, n, color in ((np.greater_equal, 5, '#2ca02c'),
                           (np.equal, 4, '#98df8a'),
                           (np.equal, 3, (1, 1, 0)),
                           (np.less_equal, 2, (1, 0, 0))):
        mask = comp(counts, n)
        mask[:-1] |= comp(counts[1:], n)
        ax.fill_between(t, 0, n_coils, where=mask,
                        color=color, edgecolor='none', linewidth=0, zorder=1)
    ax.grid(True)
    fig.tight_layout()
    plt_show(show)
    return fig


@contextmanager
def mlab_offscreen(offscreen=True):
    from mayavi import mlab
    old_offscreen = mlab.options.offscreen
    mlab.options.offscreen = offscreen
    try:
        yield
    finally:
        mlab.options.offscreen = old_offscreen


def discretize_cmap(colormap, lims, transparent=True):
    """Discretize a colormap."""
    lims = np.array(lims, int)
    assert lims.shape == (2,)
    from matplotlib import colors, pyplot as plt
    n_pts = lims[1] - lims[0] + 1
    assert n_pts > 0
    if n_pts == 1:
        vals = np.ones(256)
    else:
        vals = np.round(np.linspace(-0.5, n_pts - 0.5, 256)) / (n_pts - 1)
    colormap = plt.get_cmap(colormap)(vals)
    if transparent:
        colormap[:, 3] = np.clip((vals + 0.5 / n_pts) * 2, 0, 1)
    colormap[0, 3] = 0.
    colormap = colors.ListedColormap(colormap)
    use_lims = [lims[0] - 0.5, (lims[0] + lims[1]) / 2., lims[1] + 0.5]
    return colormap, use_lims


def trim_bg(img, color=None):
    """Trim background rows/cols from an image-like object."""
    if color is None:
        color = img[0, 0]
    img = img[:, (img != color).any(0).any(-1)]
    img = img[(img != color).any(1).any(-1)]
    return img
