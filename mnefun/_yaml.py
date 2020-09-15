from collections import defaultdict
import os.path as op
import sys

_CANONICAL_YAML_FNAME = op.join(op.dirname(__file__), 'data', 'canonical.yml')

_EXCLUDED_KEYS = (
    # Should be defined per machine
    'sws_dir',
    'sws_ssh',
    'sws_port',
    # methods
    'freeze',
    'unfreeze',
    'convert_subjects',
    'save',
    # Properties
    'report_params',
    'pca_extra',
    'pca_fif_tag',
)

_REPORT_KEYS = {
    'chpi_snr', 'good_hpi_count', 'head_movement', 'raw_segments',
    'psd', 'ssp_topomaps', 'source_alignment', 'drop_log', 'bem', 'covariance',
    'snr', 'whitening', 'sensor', 'source', 'pre_fun', 'post_fun', 'preload',
}

_KEYS_TO_FLATTEN = {
    'general', 'naming', 'fetch_raw', 'scoring', 'bads', 'raw', 'annotations',
    'multithreading',
    'preprocessing', 'head_position_estimation', 'sss', 'filtering', 'ssp',
    'epochs', 'epoching',
    'covariance', 'forward', 'inverse',
}


def _get_params_keys(p):
    from ._mnefun import Params
    assert isinstance(p, Params)
    key_set = set(key for key in dir(p) if not key.startswith('_'))
    key_set = key_set - set(_EXCLUDED_KEYS)
    return key_set


def _constant_factory(value):
    return lambda: value


def read_params(fname):
    """Create Params from a YAML file.

    Parameters
    ----------
    fname : str
        The filename to use.

    Returns
    -------
    params : instance of Params
        The parameters.
    """
    from . import Params
    params = Params()
    attrs = _flat_params_read(fname)
    for key, val in attrs.items():
        if isinstance(val, dict) and '__default__' in val:
            default = val.pop('__default__')
            val = defaultdict(_constant_factory(default), **val)
        setattr(params, key, val)
    return params


def _write_params(fname, p):
    import yaml
    from ._mnefun import Params
    def_params = Params()
    assert isinstance(p, Params)
    with open(_CANONICAL_YAML_FNAME, 'r') as fid:
        out = yaml.load(fid, Loader=yaml.SafeLoader)
    # create reverse lookup for speed
    lookup = dict()
    _flat_params_map(out, lookup)
    for key in _get_params_keys(p):
        # otherwise we have some major config issue
        assert key in lookup, f'{key} missing from {_CANONICAL_YAML_FNAME}'
        use = out
        for level in lookup[key][:-1]:
            use = use[level]
        val = _yamlize(getattr(p, key))
        if val != getattr(def_params, key):
            use[lookup[key][-1]] = val
        else:
            del use[lookup[key][-1]]
    for ii, key in enumerate(out.keys()):
        if ii != 0:
            break
        assert key == 'general'
    with open(fname, 'w') as fid:
        yaml.dump(out, fid, default_flow_style=True,
                  sort_keys=sys.version_info < (3, 7))


def _yamlize(obj):
    if isinstance(obj, tuple):
        obj = list(obj)
    elif isinstance(obj, defaultdict):
        obj = obj.copy()
        obj['__default__']  # causes the default value to be created here
        obj = dict(obj)
    return obj


def _flat_params_map(d, lookup, prefix=list()):
    assert isinstance(d, dict)
    for key, value in d.items():
        if key != 'report' and isinstance(d[key], dict):
            _flat_params_map(d[key], lookup, prefix + [key])
        else:
            lookup[key] = prefix + [key]


def _flat_params_read(fname):
    import yaml
    with open(fname, 'r') as fid:
        yvals = yaml.load(fid, Loader=yaml.SafeLoader)
    # Need to flatten
    out = dict()
    _flatten_dicts(yvals, out, _KEYS_TO_FLATTEN)
    # Set work_dir properly
    out['work_dir'] = op.abspath(out.get('work_dir', '.'))
    return out


def _flatten_dicts(d, out, keys_to_flatten):
    assert isinstance(d, dict), type(d)
    assert isinstance(out, dict)
    for key, val in d.items():
        if key in keys_to_flatten:
            _flatten_dicts(val, out, keys_to_flatten)
        else:
            assert key not in out
            out[key] = val


# Equivalence tested in `funloc` with::
#
#     import mne
#     import h5io
#     p = h5io.read_hdf5('params_dict.h5')
#     for d in (p, params.__dict__):
#         for key, val in d.items():
#             if isinstance(val, tuple):
#                 d[key] = list(val)
#     params.score = None
#     print(mne.utils.object_diff(p, params.__dict__))
#
# The only remaining differences were with array vs list-of-int.
