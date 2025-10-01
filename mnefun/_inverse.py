"""Inverse and source space data processing."""

import os
import os.path as op

import numpy as np
from mne import (read_label, read_labels_from_annot, read_source_spaces, Label,
                 SourceEstimate, BiHemiLabel, read_surface, read_epochs,
                 read_cov, read_forward_solution, convert_forward_solution,
                 pick_types_forward)
from mne.cov import regularize
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.utils import get_subjects_dir, verbose, logger

from ._cov import _compute_rank
from ._paths import (get_epochs_evokeds_fnames, safe_inserter,
                     get_cov_fwd_inv_fnames)
from ._utils import _overwrite

try:
    from mne import spatial_src_adjacency
except ImportError:
    from mne import spatial_src_connectivity as spatial_src_adjacency


def gen_inverses(p, subjects, run_indices):
    """Generate inverses.

    Can only complete successfully following forward solution
    calculation and covariance estimation.

    Parameters
    ----------
    p : instance of Parameters
        Analysis parameters.
    subjects : list of str
        Subject names to analyze (e.g., ['Eric_SoP_001', ...]).
    run_indices : array-like | None
        Run indices to include.
    """
    for si, subj in enumerate(subjects):
        out_flags, meg_bools, eeg_bools = [], [], []
        if p.disp_files:
            print('  Subject %s' % subj, end='')
        inv_dir = op.join(p.work_dir, subj, p.inverse_dir)
        cov_dir = op.join(p.work_dir, subj, p.cov_dir)
        if not op.isdir(inv_dir):
            os.mkdir(inv_dir)
        make_erm_inv = len(p.runs_empty) > 0

        epochs_fnames, _ = get_epochs_evokeds_fnames(p, subj, p.analyses)
        _, fif_file = epochs_fnames
        epochs = read_epochs(fif_file, preload=False)
        del epochs_fnames, fif_file

        meg, eeg = 'meg' in epochs, 'eeg' in epochs

        if meg:
            out_flags += ['-meg']
            meg_bools += [True]
            eeg_bools += [False]
        if eeg:
            out_flags += ['-eeg']
            meg_bools += [False]
            eeg_bools += [True]
        if meg and eeg:
            out_flags += ['-meg-eeg']
            meg_bools += [True]
            eeg_bools += [True]
        if p.cov_rank == 'full' and p.compute_rank:
            rank = _compute_rank(p, subj, run_indices[si])
        else:
            rank = None  # should be safe from our gen_covariances step
        if make_erm_inv:
            # We now process the empty room with "movement
            # compensation" so it should get the same rank!
            erm_name = op.join(cov_dir, safe_inserter(p.runs_empty[0], subj) +
                               p.pca_extra + p.inv_tag + '-cov.fif')
            empty_cov = read_cov(erm_name)
            if p.force_erm_cov_rank_full and p.cov_method == 'empirical':
                empty_cov = regularize(
                    empty_cov, epochs.info, rank='full')
        fwd_name = get_cov_fwd_inv_fnames(p, subj, run_indices[si])[1][0]
        fwd = read_forward_solution(fwd_name)
        fwd = convert_forward_solution(fwd, surf_ori=True)
        looses = [1]
        tags = [p.inv_free_tag]
        fixeds = [False]
        depths = [0.8]
        if fwd['src'].kind == 'surface':
            looses += [0, 0.2]
            tags += [p.inv_fixed_tag, p.inv_loose_tag]
            fixeds += [True, False]
            depths += [0.8, 0.8]
        else:
            assert fwd['src'].kind == 'volume'

        for name in p.inv_names + ([make_erm_inv] if make_erm_inv else []):
            if name is True:  # meaning: make empty-room one
                temp_name = subj
                cov = empty_cov
                tag = p.inv_erm_tag
            else:
                s_name = safe_inserter(name, subj)
                temp_name = s_name + ('-%d' % p.lp_cut) + p.inv_tag
                cov_name = op.join(cov_dir, safe_inserter(name, subj) +
                                   ('-%d' % p.lp_cut) + p.inv_tag + '-cov.fif')
                cov = read_cov(cov_name)
                if cov.get('method', 'empirical') == 'empirical':
                    cov = regularize(cov, epochs.info, rank=rank)
                tag = ''
                del s_name
            for f, m, e in zip(out_flags, meg_bools, eeg_bools):
                fwd_restricted = pick_types_forward(fwd, meg=m, eeg=e)
                for l_, s, x, d in zip(looses, tags, fixeds, depths):
                    inv_name = op.join(
                        inv_dir, temp_name + f + tag + s + '-inv.fif')
                    kwargs = dict(loose=l_, depth=d, fixed=x, use_cps=True,
                                  verbose='error')
                    if name is not True or not e:
                        inv = make_inverse_operator(
                            epochs.info, fwd_restricted, cov, rank=rank,
                            **kwargs)
                        _overwrite(write_inverse_operator, inv_name, inv)
        if p.disp_files:
            print()


def get_fsaverage_medial_vertices(concatenate=True, subjects_dir=None,
                                  vertices=None):
    """Return fsaverage medial wall vertex numbers.

    These refer to the standard fsaverage source space
    (with vertices from 0 to 2*10242-1).

    Parameters
    ----------
    concatenate : bool
        If True, the returned vertices will be indices into the left and right
        hemisphere that are part of the medial wall. This is
        Useful when treating the source space as a single entity (e.g.,
        during clustering).
    subjects_dir : str
        Directory containing subjects data. If None use
        the Freesurfer SUBJECTS_DIR environment variable.
    vertices : None | list
        Can be None to use ``[np.arange(10242)] * 2``.

    Returns
    -------
    vertices : list of array, or array
        The medial wall vertices.
    """
    if vertices is None:
        vertices = [np.arange(10242), np.arange(10242)]
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    label_dir = op.join(subjects_dir, 'fsaverage', 'label')
    lh = read_label(op.join(label_dir, 'lh.Medial_wall.label'))
    rh = read_label(op.join(label_dir, 'rh.Medial_wall.label'))
    if concatenate:
        bad_left = np.where(np.in1d(vertices[0], lh.vertices))[0]
        bad_right = np.where(np.in1d(vertices[1], rh.vertices))[0]
        return np.concatenate((bad_left, bad_right + len(vertices[0])))
    else:
        return [lh.vertices, rh.vertices]


@verbose
def get_fsaverage_label_operator(parc='aparc.a2009s', remove_bads=True,
                                 combine_medial=False, return_labels=False,
                                 subjects_dir=None, verbose=None):
    """Get a label operator matrix for fsaverage."""
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    src = read_source_spaces(op.join(
        subjects_dir, 'fsaverage', 'bem', 'fsaverage-5-src.fif'),
        verbose=False)
    fs_vertices = [np.arange(10242), np.arange(10242)]
    assert all(np.array_equal(a['vertno'], b)
               for a, b in zip(src, fs_vertices))
    labels = read_labels_from_annot('fsaverage', parc)
    # Remove bad labels
    if remove_bads:
        bads = get_fsaverage_medial_vertices(False)
        bads = dict(lh=bads[0], rh=bads[1])
        assert all(b.size > 1 for b in bads.values())
        labels = [label for label in labels
                  if np.in1d(label.vertices, bads[label.hemi]).mean() < 0.8]
        del bads
    if combine_medial:
        labels = combine_medial_labels(labels)
    offsets = dict(lh=0, rh=10242)
    rev_op = np.zeros((20484, len(labels)))
    for li, label in enumerate(labels):
        if isinstance(label, BiHemiLabel):
            use_labels = [label.lh, label.rh]
        else:
            use_labels = [label]
        for ll in use_labels:
            rev_op[ll.get_vertices_used() + offsets[ll.hemi], li:li + 1] = 1.
    # every src vertex is in exactly one label, except medial wall verts
    # assert (rev_op.sum(-1) == 1).sum()
    label_op = SourceEstimate(np.eye(20484), fs_vertices, 0, 1)
    label_op = label_op.extract_label_time_course(labels, src)
    out = (label_op, rev_op)
    if return_labels:
        out += (labels,)
    return out


@verbose
def combine_medial_labels(labels, subject='fsaverage', surf='white',
                          dist_limit=0.02, subjects_dir=None):
    """Combine medial labels."""
    from mne.surface import _compute_nearest
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    rrs = dict((hemi, read_surface(op.join(subjects_dir, subject, 'surf',
                                           '%s.%s'
                                           % (hemi, surf)))[0] / 1000.)
               for hemi in ('lh', 'rh'))
    use_labels = list()
    used = np.zeros(len(labels), bool)
    logger.info('Matching medial regions for %s labels on %s %s, d=%0.1f mm'
                % (len(labels), subject, surf, 1000 * dist_limit))
    for li1, l1 in enumerate(labels):
        if used[li1]:
            continue
        used[li1] = True
        use_label = l1.copy()
        rr1 = rrs[l1.hemi][l1.vertices]
        for li2 in np.where(~used)[0]:
            l2 = labels[li2]
            same_name = (l2.name.replace(l2.hemi, '') ==
                         l1.name.replace(l1.hemi, ''))
            if l2.hemi != l1.hemi and same_name:
                rr2 = rrs[l2.hemi][l2.vertices]
                mean_min = np.mean(_compute_nearest(
                    rr1, rr2, return_dists=True)[1])
                if mean_min <= dist_limit:
                    use_label += l2
                    used[li2] = True
                    logger.info('  Matched: ' + l1.name)
        use_labels.append(use_label)
    logger.info('Total %d labels' % (len(use_labels),))
    return use_labels


def get_hcpmmp_mapping():
    """Get the Glasser number : name mapping.

    Returns
    -------
    mapping : dict
        The Glasser number : name mapping. In principle this could be
        a list that can be used with `enumerate` and +1, but people
        usually give regions via numbers so this should be most convenient.
    """
    # https://images.nature.com/full/nature-assets/nature/journal/v536/n7615/extref/nature18933-s3.pdf  # noqa
    return {ii + 1: key for ii, key in enumerate([
        'Primary Visual Cortex (V1)',  # 1
        'Early Visual Cortex',  # 2
        'Dorsal Stream Visual Cortex',  # 3
        'Ventral Stream Visual Cortex',  # 4
        'MT+ Complex and Neighboring Visual Areas',  # 5
        'Somatosensory and Motor Cortex',  # 6
        'Paracentral Lobular and Mid Cingulate Cortex',  # 7
        'Premotor Cortex',  # 8
        'Posterior Opercular Cortex',  # 9
        'Early Auditory Cortex',  # 10
        'Auditory Association Cortex',  # 11
        'Insular and Frontal Opercular Cortex',  # 12
        'Medial Temporal Cortex',  # 13
        'Lateral Temporal Cortex',  # 14
        'Temporo-Parieto-Occipital Junction',  # 15
        'Superior Parietal Cortex',  # 16
        'Inferior Parietal Cortex',  # 17
        'Posterior Cingulate Cortex',  # 18
        'Anterior Cingulate and Medial Prefrontal Cortex',  # 19
        'Orbital and Polar Frontal Cortex',  # 20
        'Inferior Frontal Cortex',  # 21
        'DorsoLateral Prefrontal Cortex',  # 22
        '???',  # 23
    ])}


def extract_roi(stc, src, label=None, thresh=0.5):
    """Extract a functional ROI.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source estimate data. The maximum positive peak will be selected.
        If you want the maximum negative peak, consider passing
        abs(stc) or -stc.
    src : instance of SourceSpaces
        The associated source space.
    label : instance of Label | None
        The label within which to select the peak.
        Can be None to use the entire STC.
    thresh : float
        Threshold value (relative to the peak value) above which vertices
        will be taken.

    Returns
    -------
    roi : instance of Label
        The functional ROI.
    """
    assert isinstance(stc, SourceEstimate)
    if label is None:
        stc_label = stc.copy()
    else:
        stc_label = stc.in_label(label)
    del label
    max_vidx, max_tidx = np.unravel_index(np.argmax(stc_label.data),
                                          stc_label.data.shape)
    max_val = stc_label.data[max_vidx, max_tidx]
    if max_vidx < len(stc_label.vertices[0]):
        hemi = 'lh'
        max_vert = stc_label.vertices[0][max_vidx]
        max_vidx = list(stc.vertices[0]).index(max_vert)
    else:
        hemi = 'rh'
        max_vert = stc_label.vertices[1][max_vidx - len(stc_label.vertices[0])]
        max_vidx = list(stc.vertices[1]).index(max_vert)
        max_vidx += len(stc.vertices[0])
    del stc_label
    assert max_val == stc.data[max_vidx, max_tidx]

    # Get contiguous vertices within 50%
    threshold = max_val * thresh
    connectivity = spatial_src_adjacency(src, verbose='error')  # holes
    _, clusters, _, _ = spatio_temporal_cluster_1samp_test(
        np.array([stc.data]), threshold, n_permutations=1,
        stat_fun=lambda x: x.mean(0), tail=1,
        connectivity=connectivity)
    for cluster in clusters:
        if max_vidx in cluster[0] and max_tidx in cluster[1]:
            break  # found our cluster
    else:  # in case we did not "break"
        raise RuntimeError('Clustering failed somehow!')
    if hemi == 'lh':
        verts = stc.vertices[0][cluster]
    else:
        verts = stc.vertices[1][cluster - len(stc.vertices[0])]
    func_label = Label(verts, hemi=hemi, subject=stc.subject)
    func_label = func_label.fill(src)
    return func_label, max_vert, max_vidx, max_tidx
