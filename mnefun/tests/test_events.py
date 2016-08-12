from mock import Mock, patch
import mnefun


def test_event():
    """Test that on_process event gets fired"""
    handler = Mock()
    params = mnefun.Params()
    setdefaults(params)
    params.on_process = handler
    with patch('mnefun._mnefun.inspect'):
        with patch('mnefun._mnefun.save_epochs') as func:
            mnefun.do_processing(params,
                                 write_epochs=True,
                                 print_status=False)
            handler.assert_called_with('Doing epoch EQ/DQ',
                                       func, func(), params)


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
