import os.path as op
import yaml

import pytest
from numpydoc import docscrape
from mnefun import Params


_excluded_keys = (
    'acq_dir',
    'acq_ssh',
    'acq_port',
    'sws_dir',
    'sws_ssh',
    'sws_port',
)


def test_params(params):
    """Test that all of our Params options are documented in the docstring."""
    # class doc has all keys in attributes
    key_set = set(key for key in dir(params) if not key.startswith('_'))
    key_set = key_set - set(_excluded_keys)
    doc = docscrape.ClassDoc(Params)
    doc = [p.name for p in doc['Attributes']]
    assert set(doc) == key_set

    # funloc documented the same way
    fname = op.join(op.dirname(__file__), '..', '..', 'examples', 'funloc',
                    'funloc_params.yml')
    assert op.isfile(fname)  # only works on dev-installed mnefun
    with open(fname, 'r') as fid:
        yvals = yaml.safe_load(fid)
    assert set(yvals) == key_set

    # on Python3.7 we are guaranteed insertion order, so this should be okay
    assert list(yvals.keys()) == doc
