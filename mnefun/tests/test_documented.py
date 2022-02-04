import os.path as op
import re

from mnefun import Params
from mnefun._yaml import (_flat_params_read, _CANONICAL_YAML_FNAME,
                          _REPORT_KEYS, _get_params_keys)


def test_params(params):
    """Test that all of our Params options are documented in the docstring."""
    # class doc has all keys in attributes
    key_set = _get_params_keys(params)
    p = Params()
    assert set(p.report_params.keys()) == set(_REPORT_KEYS)
    fname = op.join(op.dirname(__file__), '..', '..', 'doc', 'overview.rst')
    with open(fname, 'r') as fid:
        f = fid.read()
    attrs = re.findall('(\\S+) : .*\n(?:    \\S+.+\n)+', f, re.MULTILINE)
    for key in _REPORT_KEYS:
        assert key in attrs
        attrs.pop(attrs.index(key))
    attrs.insert(attrs.index('list_dir'), 'report')
    assert set(attrs) == key_set, \
        'doc/overview.rst mismatch <->Params attribute'

    # canonical document
    yvals = _flat_params_read(_CANONICAL_YAML_FNAME)
    # on Python3.7 we are guaranteed insertion order, so this should be okay
    yvals = list(yvals.keys())
    assert set(yvals) == key_set, 'Mismatch canonical<->Params'
    assert yvals == attrs, 'Canonical yaml <-> Params attribute mismatch'

    # funloc has a correct subset
    fname = op.join(op.dirname(__file__), '..', '..', 'examples', 'funloc',
                    'funloc_params.yml')
    assert op.isfile(fname)  # only works on dev-installed mnefun
    yvals = _flat_params_read(fname)
    # on Python3.7 we are guaranteed insertion order, so this should be okay
    assert set(yvals) - key_set == set()
    assert key_set - set(yvals) != set()
