What's new
==========

.. currentmodule:: mnefun

Using the latest features typically require being on an up-to-date ``mnefun``
master branch and often an up-to-date MNE-Python ``master`` branch.

Changelog
~~~~~~~~~

2020
^^^^
- 2020/09/28:
    Added :func:`mnefun.make_dipole_projectors`.
- 2020/09/17:
    Added support for HEOG and VEOG projectors via additional
    projection options.
- 2020/09/16:
    Added support for the following parameters to take dict (including
    defaultdict) so that subject-specific parameters can be used:

    - ``coil_dist_limit``
    - ``coil_gof_limit``
    - ``coil_t_step_min``
    - ``coil_t_window``
    - ``rotation_limit``
    - ``translation_limit``
    - ``coil_bad_count_duration_limit``
- 2020/09/15:
    Added support for ``params.filter_chpi_t_window`` for backward
    compatibility with when this was fixed to 0.2 (around Feb 2020).
- 2020/08/26:
    Optimized window length for cHPI amplitude estimation by accounting for
    line frequency and its harmonics. The window for a standard minimum cHPI
    frequency of 83 Hz should now be longer (determined by the 60 Hz line
    component) making head position estimation more robust.
- 2020/07/21:
    Added support for ``cont_hp`` to allow high-pass filtering (in addition
    to existing low-pass filtering via ``cont_lp``; thereby allowing band-pass
    filtering) when computing empty-room projectors.
- 2020/07/16:
    Added support for ``epochs_proj='delayed'`` and
    ``report_params['sensor'] = dict(..., proj=False)`` to allow plotting
    sensor space responses without projection.
- 2020/07/08:
    Add support for ``pre_fun`` and ``post_fun`` hooks in reports.
- 2020/06/19
    Fixed errant picking of only data channels in preprocessed raw data files.
- 2020/05/26
    Improved :class:`mne.Report` generation for comparing conditions by using
    sliders and grouping by sensor types.
- 2020/04/28
    Added ``every_other`` support for computing evoked data with
    even and odd trials.
- 2020/04/01
    Added peak-detection capability to reports for the sensor and source
    sections, using ``times='peaks'``. Peaks are based on whitened gfps.
- 2020/03/06
    Added Python-based Maxwell-filter automatic bad channel detection
    using ``mf_autobad_type='python'``.
- 2020/02/21
    Added Python-based head position estimation using ``hp_type='python'``.
- 2020/02/17
    Added automated QA application ``acq_qa`` to monitor
    acquisition directories and generate HTML reports.

2019
^^^^
- 2019/12/02
    Added :func:`get_atlas_roi_mask` and :func:`get_atlas_mapping`
    to aid in extracting volumetric ROIs until it's formalized in MNE.
- 2019/12/01
    Added :func:`read_params` and YAML support for :class:`Params`.

Prior
^^^^^

- 2015-2019/12/01
    A whole bunch of undocumented updates.
- 2015/05/01
    Added simulation code.
- 2015/05/29
    Added spherical model support.
