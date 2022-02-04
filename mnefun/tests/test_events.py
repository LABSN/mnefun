from unittest.mock import Mock, patch
import mnefun


def test_event(params):
    """Test that on_process event gets fired"""
    handler = Mock()
    params.on_process = handler
    with patch('mnefun._mnefun.save_epochs') as func:
        mnefun.do_processing(params, write_epochs=True,
                             print_status=False)
        handler.assert_called_with('Doing epoch EQ/DQ',
                                   func, func(), params)
