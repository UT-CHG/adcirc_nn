#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
Context module for adding system path for unit tests.
"""
import os
import sys
adcirc_nn_root=os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                    )
                )
            )
        )
sys.path.insert(0, os.path.abspath(adcirc_nn_root))

