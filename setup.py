#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from distutils.command.install import INSTALL_SCHEMES
from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'vtem'
VERSION =            '0.1.2.1'
AUTHOR =             'Yiyin Zhou'
AUTHOR_EMAIL =       'yz2227@columbia.edu'
URL =                'https://github.com/bionet/vtem/'
DESCRIPTION =        'Video Time Encoding and Decoding Machines'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering']
PACKAGES =           ['vtem','vtem.utils']

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        packages = PACKAGES,
        install_requires = ['scikits.cuda >= 0.042',
                            'pycuda >= 2011.1',
                            'tables >= 2.4.0',
                            'numexpr >= 2.0.0',
                            'numpy >= 1.6.0']
        )
