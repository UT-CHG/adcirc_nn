#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
Module template for testing.
"""
import unittest
import numpy as np

if __name__ == '__main__':
    # Executing test directly over command line
    from adcirc_nn_unitcontext import *
else:
    # Executing test as module
    from .adcirc_nn_unitcontext import *

#------------------------------------------------------------------------------#
LOCALDEBUG = 0

#------------------------------------------------------------------------------#
class TestUnittestTemplate(unittest.TestCase):
    """Unit test class template."""

    def setUp(self):
        """Initialize the unit test."""
        pass

    def tearDown(self):
        """Finalize the unit test."""
        pass

    def test_main(self):
        """Run the unit test."""
        pass

#------------------------------------------------------------------------------#
if __name__ == '__main__':
    unittest.main()

