import numpy as np
from nose.tools import raises, eq_


# decorator - testing assertion statement
@raises(AssertionError)
def test_get_ave_trans_assert_error():
    from mnefun._mnefun import _get_ave_trans
    pos = np.random.rand(20, 9)
    _get_ave_trans(pos)


def test_get_ave_trans():
    """Test that _get_ave_trans returns 4x4 array"""
    from mnefun._mnefun import _get_ave_trans
    # mock input
    pos = np.ones((20, 10))
    # run func with input
    out = _get_ave_trans(pos)
    # test output
    eq_(out.shape, (4, 4))
