# -*- coding: utf-8 -*-

from ._atlas import get_atlas_mapping, get_atlas_roi_mask
from ._paths import get_raw_fnames, get_event_fnames
from ._fix import ch_names_uw_60, ch_names_uw_70, fix_eeg_channels
from ._mnefun import Params, do_processing
from ._inverse import get_fsaverage_medial_vertices, \
    get_fsaverage_label_operator, combine_medial_labels, get_hcpmmp_mapping, \
    extract_roi
from ._paths import safe_inserter
from ._scoring import extract_expyfun_events
from ._sss import run_sss_command, run_sss_positions, info_sss_basis, \
    compute_good_coils
from ._viz import plot_reconstruction, plot_chpi_snr_raw, plot_good_coils, \
    clean_brain, plot_colorbar, discretize_cmap
from ._utils import make_montage, compute_auc
from ._stats import anova_time
from ._viz import trim_bg
from ._yaml import read_params

__version__ = '0.2.0.dev0'
