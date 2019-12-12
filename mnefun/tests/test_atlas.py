import pytest
from mnefun import get_atlas_mapping


@pytest.mark.parametrize('atlas, count, label_21', [
    ('LBPA40', 56, 'L_superior_frontal_gyrus'),
    ('IXI', 83, 'Insula_right'),
])
def test_get_atlas_mapping(atlas, count, label_21):
    """Test getting atlas mappings."""
    mapping = get_atlas_mapping(atlas)
    assert label_21 in mapping
    assert mapping[label_21] == 21
    assert len(mapping) == count
