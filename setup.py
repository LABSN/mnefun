#! /usr/bin/env python

import os

import setuptools  # noqa : we are using a setuptools namespace
from numpy.distutils.core import setup

version = "0.1"
descr = """Methods for integrating LABSN with mne-python"""

DISTNAME = 'mnefun'
DESCRIPTION = descr
MAINTAINER = 'Eric Larson'
MAINTAINER_EMAIL = 'larsoner@uw.edu'
URL = 'http://github.com/LABSN/mnefun'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/LABSN/mnefun'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[],
          platforms='any',
          packages=setuptools.find_packages(),
          package_data={'mnefun': ['run_sss.sh', os.path.join('data', 'sss_cal.dat'),
                                   os.path.join('data', 'ct_sparse.fif')]},
          scripts=['bin/simulate_movement.py', 'bin/plot_chpi_snr.py',
                   'bin/chpi_weighted_average.py'])
