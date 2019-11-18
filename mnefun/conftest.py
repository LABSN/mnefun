from unittest.mock import Mock
import pytest
import mnefun


@pytest.fixture(scope='function')
def params(tmpdir):
    """Param fixture."""
    params = mnefun.Params()
    params.work_dir = str(tmpdir)
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
    return params
