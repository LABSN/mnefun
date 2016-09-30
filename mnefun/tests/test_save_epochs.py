from mock import Mock, patch
import mnefun
import numpy
from nose.tools import assert_raises, assert_in


def test_exits_with_warning_if_all_epochs_autorejected():
    """Test that save_epochs fails gracefully if no epochs remain."""
    params = mnefun.Params()
    setdefaults(params)
    params.subject_indices = numpy.arange(1)
    params.subjects = ['x']
    params.run_names = ['%s']
    params.analyses = ['x']
    params.out_names = [[]]
    params.out_numbers = [[]]
    params.must_match = [[]]
    with patch('mnefun._mnefun.inspect'):
        with patch('mnefun._mnefun.os'):
            with patch('mnefun._mnefun.Raw'):
                with patch('mnefun._mnefun.concatenate_events'):
                    with patch('mnefun._mnefun._fix_raw_eog_cals'):
                        with patch('mnefun._mnefun.get_event_fnames'):
                            with patch('mnefun._mnefun.Epochs') as Epochs:
                                epochs = Mock()
                                epochs.info = {'sfreq': 1000}
                                epochs.events = numpy.zeros((0, 3))
                                Epochs.return_value = epochs
                                with assert_raises(ValueError) as cm:
                                        mnefun.do_processing(params,
                                                             write_epochs=True,
                                                             print_status=False)
                                assert_in('No valid epochs', str(cm.exception))


def setdefaults(params):
    params.score = Mock()
    params.subject_indices = []
    params.subjects = [None]
    params.structurals = [None]
    params.dates = [None]
    params.in_names = []
    params.in_numbers = []
    params.analyses = []
    params.out_names = []
    params.out_numbers = []
    params.must_match = []
    params.decim = 5
    params.plot_drop_logs = False
