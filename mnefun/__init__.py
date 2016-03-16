# -*- coding: utf-8 -*-

from ._paths import get_raw_fnames, get_event_fnames  # noqa
from ._mnefun import (Params, get_fsaverage_medial_vertices,  # noqa
                      safe_inserter,  # noqa
                      extract_expyfun_events, do_processing,  # noqa
                      run_sss_command, run_sss_positions)  # noqa
from .stats import (anova_time, descriptives)

__version__ = '0.1.0.dev0'
