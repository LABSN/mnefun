import os
import os.path as op
from unittest.mock import Mock, patch

import pytest
import numpy as np

import mnefun


def test_exits_with_warning_if_all_epochs_autorejected(params):
    """Test that save_epochs fails gracefully if no epochs remain."""
    params.subject_indices = np.arange(1)
    params.subjects = ['x']
    params.run_names = ['%s']
    params.analyses = ['x']
    params.out_names = [[]]
    params.out_numbers = [[]]
    params.must_match = [[]]
    os.makedirs(op.join(params.work_dir, params.subjects[0], 'sss_pca_fif'))
    with patch('mnefun._epoching.read_raw_fif') as read_raw_fif, \
            patch('mnefun._scoring.concatenate_events'), \
            patch('mnefun._epoching._fix_raw_eog_cals'), \
            patch('mnefun._scoring.get_event_fnames') as get_event_fnames, \
            patch('mnefun._epoching._read_events') as _read_events, \
            patch('mnefun._epoching.Epochs') as Epochs:
        Raw = Mock()
        Raw.info = {'sfreq': 1000}
        Raw._first_samps = [0]
        Raw._last_samps = [1000]
        read_raw_fif.return_value = Raw
        get_event_fnames.return_value = ['foo']
        _read_events.return_value = np.empty((0, 3), int)
        epochs = Mock()
        epochs.info = {'sfreq': 1000}
        epochs.events = np.zeros((0, 3))
        Epochs.return_value = epochs
        with pytest.raises(ValueError, match='No valid'):
            mnefun.do_processing(params, write_epochs=True,
                                 print_status=False)
