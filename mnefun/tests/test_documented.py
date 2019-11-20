import os.path as op

from numpydoc import docscrape
from mnefun import Params
from mnefun import _flat_params_read


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


def test_params(params):
    """Test that all of our Params options are documented in the docstring."""
    # class doc has all keys in attributes
    key_set = set(key for key in dir(params) if not key.startswith('_'))
    key_set = key_set - set(_EXCLUDED_KEYS)
    doc = docscrape.ClassDoc(Params)
    doc = [p.name for p in doc['Attributes']]
    assert set(doc) == key_set

    # funloc documented the same way
    fname = op.join(op.dirname(__file__), '..', '..', 'examples', 'funloc',
                    'funloc_params.yml')
    assert op.isfile(fname)  # only works on dev-installed mnefun
    yvals = _flat_params_read(fname)
    assert set(yvals) == key_set

    # on Python3.7 we are guaranteed insertion order, so this should be okay
    assert list(yvals.keys()) == doc
