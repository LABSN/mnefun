# -*- coding: utf-8 -*-

from ._paths import get_raw_fnames, get_event_fnames
from ._mnefun import (
    Params, get_fsaverage_medial_vertices, safe_inserter,
    extract_expyfun_events, do_processing, run_sss_command, run_sss_positions,
    info_sss_basis, plot_reconstruction, plot_chpi_snr_raw,
    get_fsaverage_label_operator, compute_good_coils, plot_good_coils,
    compute_auc, combine_medial_labels, clean_brain, plot_colorbar)
from stats import anova_time

__version__ = '0.1.0.dev0'
