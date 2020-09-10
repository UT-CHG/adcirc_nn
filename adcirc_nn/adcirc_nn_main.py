#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
Main function of adcirc-nn, calling inititialize, run, and finalize.
"""

from __future__ import print_function

from sys import version_info
import time
import ctypes as ct

from pyADCIRC import libadcpy as pa

if __name__ == '__main__':
    from coupler.adcirc_nn_class import AdcircNN
else:
    from .coupler.adcirc_nn_class import AdcircNN


################################################################################
DEBUG_LOCAL = 1

__all__ = ['main'] # The only thing from this module to import if needed.

################################################################################
def main():
    """Main function of adcirc-nn."""

    argv = ct.POINTER(ct.c_wchar_p if version_info >= (3, ) else ct.c_char_p)()
    argc = ct.c_int()
    ct.pythonapi.Py_GetArgcArgv(ct.byref(argc), ct.byref(argv))

    if DEBUG_LOCAL == 1:
        print('Number of arguments passed to python: {}\nArgs:'.format(argc.value)),
        for i in range(argc.value):
            print('    {}'.format(argv[i]))
        print
    if (argc.value < 4):
        print("\nProblem with command line arguments.")
        print("Format is : python <*.py> <Coupling type> <ADCIRC edge string ID>")
        print("Coupling type is one of {Adn, ndA, AdndA, ndAdn}")
        print("Exiting without testing.")
        return -1

    print("Coupling type: {}".format(argv[argc.value-2]))
    print("ADCIRC boundary string ID: {}".format(argv[argc.value-1]))

    t0 = time.time()
    print("Initializing adcirc-nn")
    adcnn = AdcircNN()
    adcnn.coupler_initialize(argc, argv)

    t1 = time.time()
    print("Running adcirc-nn")
    adcnn.coupler_run()

    t2 = time.time()
    print("Finalizing adcirc-nn")
    adcnn.coupler_finalize()

    t3 = time.time()

    tInit = t1-t0
    tRun = t2-t1
    tFin = t3-t2
    tTot = t3-t0

    print("Initialize time = {0}".format(tInit))
    print("Run time        = {0}".format(tRun))
    print("Finalize time   = {0}".format(tFin))
    print("Total time      = {0}".format(tTot))

    print("\nFinished running adcirc-nn")

    return 0

################################################################################
if __name__ == '__main__':
    main()
