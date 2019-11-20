import os.path as op


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
        setattr(params, key, val)
    return params


_KEYS_TO_FLATTEN = {
    'paths', 'fetch_raw', 'scoring', 'bads', 'raw', 'annotations',
    'multithreading',
    'preprocessing', 'head_position_estimation', 'sss', 'filtering', 'ssp',
    'epochs', 'epoching',
    'covariance', 'forward', 'inverse',
}


def _flat_params_read(fname):
    import yaml
    with open(fname, 'r') as fid:
        yvals = yaml.load(fid, Loader=yaml.SafeLoader)
    # Need to flatten
    out = dict()
    _flatten_dicts(yvals, out, _KEYS_TO_FLATTEN)
    # Change report->report_params
    if 'report' in out:
        out['report_params'] = out['report']
        del out['report']
    # Set work_dir properly
    out['work_dir'] = op.abspath(out['work_dir'])
    return out


def _flatten_dicts(d, out, keys_to_flatten):
    assert isinstance(d, dict)
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
