import pytest

from mnefun._fix import (ch_names_mgh60, ch_names_mgh70,
                         ch_names_uw_70, ch_names_uw_60,
                         ch_names_1020, ch_names_32)


@pytest.mark.parametrize(
    'check', (ch_names_mgh60, ch_names_mgh70,
              ch_names_uw_70, ch_names_uw_60))
def test_montage_defs(check):
    """Test our montage channel defs."""
    assert len(ch_names_uw_70) == len(ch_names_uw_60) == 60
    assert len(ch_names_mgh70) == 70
    assert len(ch_names_mgh60) == 60
    assert len(ch_names_1020) == 21
    assert len(ch_names_32) == 32

    miss_32 = set(ch_names_32) - set(check)
    miss_1020 = set(ch_names_1020) - set(check)
    if check in (ch_names_uw_70, ch_names_mgh70):
        assert miss_32 == set()
    else:
        assert len(miss_32) == 2
    assert miss_1020 == set()
