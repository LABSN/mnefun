def gravrevokeds(directory, subjects, analysis, condition, filtering,
                 baseline=(None, 0)):
    """helper for creating group averaged evoked file

    Parameters
    ----------
    directory : str
        MNEFUN parent study database directory.
    subjects : list
        List of subjects to combine evoked data across.
    analysis : str
        Evoked data set name.
    condition : str
        Evoked condition.
    filtering : int
        Low pass filter setting used to create evoked files.
    baseline : None | tuple
        The time interval to apply baseline correction. If None do not apply it.
        If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None
        then b is set to the end of the interval. If baseline is equal
        to (None, None) all the time interval is used.
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
