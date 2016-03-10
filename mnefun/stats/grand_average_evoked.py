def _gravrevokeds(directory, subjects, analysis, condition, filtering,
                  baseline=(None, 0)):
    """helper for creating group averaged evoked file

    Parameters
    ----------
    directory : str
    subjects : list
    analysis : str
    condition : str
    filtering : int
    """

    from os import path as op
    from mne import (read_evokeds, grand_average)

    evokeds = []
    for subj in subjects:
        evoked_file = op.join(directory, subj, 'inverse',
                              '%s_%d-sss_eq_%s-ave.fif' % (analysis, filtering, subj))
        evokeds.append(read_evokeds(evoked_file, condition=condition, baseline=baseline))
    grandavr = grand_average(evokeds)
    return grandavr
