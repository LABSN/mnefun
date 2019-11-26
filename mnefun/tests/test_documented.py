import os.path as op
import re

from mnefun import Params
from mnefun import _flat_params_read
import yaml


_EXCLUDED_KEYS = (
    # Should be defined per machine
    'sws_dir',
    'sws_ssh',
    'sws_port',
    # Actually methods
    'freeze',
    'unfreeze',
    'convert_subjects',
    # Properties
    'pca_extra',
    'pca_fif_tag',
)
_EXCLUDE_REPORT = {
    'chpi_snr', 'good_hpi_count', 'head_movement', 'raw_segments',
    'psd', 'ssp_topomaps', 'source_alignment', 'drop_log', 'bem', 'covariance',
    'snr', 'whitening', 'sensor', 'source',
}


def test_params(params):
    """Test that all of our Params options are documented in the docstring."""
    # class doc has all keys in attributes
    key_set = set(key for key in dir(params) if not key.startswith('_'))
    key_set = key_set - set(_EXCLUDED_KEYS)
    fname = op.join(op.dirname(__file__), '..', '..', 'doc', 'overview.rst')
    with open(fname, 'r') as fid:
        f = fid.read()
    attrs = re.findall('(\\S+) : .*\n(?:    \\S+.+\n)+', f, re.MULTILINE)
    del f
    assert set(attrs) - _EXCLUDE_REPORT == key_set

    # funloc documented the same way
    fname = op.join(op.dirname(__file__), '..', '..', 'examples', 'funloc',
                    'funloc_params.yml')
    assert op.isfile(fname)  # only works on dev-installed mnefun
    yvals = _flat_params_read(fname)
    # on Python3.7 we are guaranteed insertion order, so this should be okay
    yvals = list(yvals.keys())
    assert set(yvals) == key_set
    assert yvals == attrs
