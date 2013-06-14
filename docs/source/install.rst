.. -*- rst -*-

Installation Instructions
=========================

Quick Installation
------------------
If you have `pip <http://pypi.python.org/pypi/pip>`_ installed, you should be
able to install the latest stable release of ``vtem`` by running the
following::

   pip install vtem

All dependencies should be automatically downloaded and installed if they are
not already on your system.

Obtaining the Latest Software
-----------------------------
The latest stable and development versions of ``vtem`` can be downloaded from 
`GitHub <https://github.com/bionet/vtem/>`_

Online documentation for ``vtem`` is available at 
`<http://bionet.github.io/vtem/index.html/>`_

Installation Dependencies
-------------------------
``vtem`` requires that the following software packages be
installed:

* `Python <http://www.python.org>`_ 2.7 or later.
* `setuptools <http://peak.telecommunity.com/DevCenter/setuptools>`_ 0.6c10 or later.
* `NumPy <http://numpy.scipy.org>`_ 1.6.0 or later.
* `Scikits.cuda <http://github.com/lebedov/scikits.cuda/>`_ 0.042 or greater
* `PyTables <http://www.pytables.org/>`_ 2.4.0 or greater
* `PyCUDA <http://mathema.tician.de/software/pycuda>`_ 2011.1 or later 
* `NIVIDIA CUDA Toolkit <http://www.nvidia.com/object/cuda_home_new.html>`_ 4.0 or later.

To support reading and writing video files directly, the package requires

* `OpenCV <http://opencv.willowgarage.com/wiki/>`_ with python
  and `FFmpeg <http://www.ffmpeg.org/>`_ support
 
To build the documentation, the following packages are also required:

* `Sphinx <http://http://sphinx-doc.org/>`_ 1.1.0 or later.


Running the demos
-----------------
To run the demos in the package as it is, please download the `demo files
<http://www.bionet.ee.columbia.edu/code/vtem/demo_files>`_ and unarchive it in
the demos folder.

To run the demos on other videos, the filename can be provided with the -i
switch. All options supported can be viewed by::

    python <demo_filename> --help

If you have OpenCV installed, all video files encoded with codecs that have been
installed with OpenCV should be supported.

To run without OpenCV support, please load the video into MATLAB and save it
using the scripts provided in the matlab_h5 folder of the package.

Platform Support
----------------
The software has been developed and tested on Linux; it should also 
work on other Unix-like platforms supported by the above packages. Parts of the
package may work on Windows as well, but remain untested.

Building and Installation
-------------------------
To build and install the toolbox, download and unpack the source 
release and run::

   python setup.py install

from within the main directory in the release. To rebuild the
documentation, run::

   python setup.py build_sphinx

Getting Started
---------------
Sample codes for the package are located in the ``/demos`` subdirectory.
For a detailed description of the package modules, please refer the 
reference section of the package documentation.
