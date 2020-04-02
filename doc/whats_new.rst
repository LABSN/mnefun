What's new
==========

.. currentmodule:: mnefun

Using the latest features typically require being on an up-to-date ``mnefun``
master branch and often an up-to-date MNE-Python ``master`` branch.

Changelog
~~~~~~~~~

2020
^^^^
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
