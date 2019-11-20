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


def _flat_params_read(fname):
    import yaml
    with open(fname, 'r') as fid:
        yvals = yaml.load(fid, Loader=yaml.SafeLoader)
    # Need to flatten
    out = dict()
    dict_vals_allowed = {'src', 'proj_nums', 'reject', 'flat', 'report'}
    _flatten_dicts(yvals, out, dict_vals_allowed)
    if 'report' in out:
        out['report_params'] = out['report']
        del out['report']
    return out


def _flatten_dicts(d, out, dict_vals_allowed):
    assert isinstance(d, dict)
    assert isinstance(out, dict)
    for key, val in d.items():
        if isinstance(val, dict) and key not in dict_vals_allowed:
            _flatten_dicts(val, out, dict_vals_allowed)
        else:
            assert key not in out
            out[key] = val
