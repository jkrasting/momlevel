Installation
============

**momlevel** is built on the Xarray ecosystem of tools. Processing sea level can be computationally expensive, especially for centennial scale climate simulations and high resolution models. The use of Dask is highly recommended.

Prerequisites and dependencies
------------------------------
**momlevel** requires Python version 3.7 or greater and several additional pacakages
that are listed below:

* Python >=3.7
* nc-time-axis
* numpy
* xarray

Installing via Package Managers
-------------------------------
**momlevel** is available via PyPi and Anaconda.

To install from Anaconda:
.. parsed-literal::
   conda install -c krasting momlevel

To install from PyPi:
.. parsed-literal::
   pip install momlevel