import os.path as op
import pytest
from mne.utils import object_diff
from mnefun import read_params


funloc_fname = op.join(op.dirname(__file__), '..', '..', 'examples', 'funloc',
                       'funloc_params.yml')


@pytest.mark.parametrize('fname', (funloc_fname,))
def test_params_io(tmpdir, fname):
    """Test params I/O round-trip."""
    import yaml
    assert op.isfile(fname)
    p = read_params(fname)
    temp_fname = str(tmpdir.join('test.yml'))
    p.save(temp_fname)

    # Look at our objects
    p2 = read_params(temp_fname)
    assert object_diff(p.__dict__, p2.__dict__) == ''

    # Look at the YAML to check out write order in the file is preserved
    with open(fname, 'r') as fid:
        orig = yaml.load(fid, Loader=yaml.SafeLoader)
    for ii, key in enumerate(orig.keys()):
        if ii != 0:
            break
        assert key == 'general'

    with open(temp_fname, 'r') as fid:
        read = yaml.load(fid, Loader=yaml.SafeLoader)
    for ii, key in enumerate(read.keys()):
        if ii != 0:
            break
        assert key == 'general'

    # We can do this because we currently write only keys that differ from
    # defaults, but we need to exclude the ones from funloc that we include
    # anyway. This also helps ensure that the funloc example mostly shows
    # deviations from defaults.
    del orig['general']['subjects_dir']
    del orig['scoring']['score']
    for key in ('int_order', 'ext_order', 'tsss_dur', 'st_correlation'):
        del orig['preprocessing']['sss'][key]
    del orig['epoching']['decim']
    del orig['forward']['bem_type']
    for key in orig.keys():
        assert orig[key] == read[key]
