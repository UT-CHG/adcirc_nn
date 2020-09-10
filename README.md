# AdcircNN - Physics based machine learning model with ADCIRC

AdcircNN is a Python software for physics based machine learning with ADvanced
CIRCulation ([ADCIRC](http://adcirc.org/)) and neural networks. AdcircNN is a
sister project of
[Water Coupler](https://github.com/gajanan-choudhary/water-coupler) and may be
integrated into it later.


## Getting Started

### Dependencies

* Python 3+.
* [NumPy](https://numpy.org/)
* pyADCIRC, the Python interface of ADCIRC, which requires:
    * ADCIRC shared library   : `lib*adcpy.so*`
    * ADCIRC python interface : `pyadcirc*.so`


### Installing

Following steps are needed to properly install AdcircNN on Linux:
* Install the ADCIRC Python interface, pyADCIRC, and run tests if needed
* Add the path of the shared libraries `lib*adcpy.so*` and `pyadcirc*.so` to the
  LD\_LIBRARY\_PATH environment variable, and
* Run the following commands to clone this repository, change the directory, and
  pip-install AdcircNN.
  ```bash
  git clone https://github.com/UT-CHG/adcirc_nn.git`
  cd adcirc_nn
  python -m pip install .
  ```
* You should now be able to `import adcirc_nn` in Python, if needed.


### Running tests

To do.


## Using the project

Suppose you want to run an AdcircNN simulation. After installing pyADCIRC and
AdcircNN, copy the ADCIRC input files in a single directory. Run AdcircNN as a
module with 2 command line arguments as follows.
 - Argument 1: Coupling type identifier which is one of {Adn, ndA, AdndA, ndAdn}
 - Argument 2: Boundary string ID of the ADCIRC model that is being coupled to
   the machine learning model,
For instance, the Linux/Unix workflow goes as follows.
```bash
mkdir sample-sim
cd sample-sim
cp <path_to_ADCIRC_input_files>/fort.* .

python -m adcirc_nn <Coupling type identifier>  <ADCIRC model coupled boundary>
```


## License

AdcircNN is distributed under the
[BSD 3-Clause "New" or "Revised" license](LICENSE). Note, however, that some of
the external dependencies of this software are proprietary/closed-source, which
cannot and should not be distributed with this software.

