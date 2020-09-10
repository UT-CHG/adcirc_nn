#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
The main adcirc-nn module.

Contains the main adcirc-nn function imported as "main".
"""

if __name__ == '__main__':
    from adcirc_nn_main import main
else:
    from .adcirc_nn_main import main

