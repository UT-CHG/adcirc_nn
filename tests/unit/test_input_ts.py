#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
Module for testing the InputTS class.
"""
import unittest
import numpy as np

if __name__ == '__main__':
    # Executing test directly over command line
    from adcirc_nn_unitcontext import *
else:
    # Executing test as module
    from .adcirc_nn_unitcontext import *

from adcirc_nn.coupler.input_ts import InputTS
from pyADCIRC import pyadcirc_path
from pyADCIRC import libadcpy as pa

#------------------------------------------------------------------------------#
LOCALDEBUG = 0

#------------------------------------------------------------------------------#
class TestInputTS(unittest.TestCase):
    """Unit test for InputTS class."""

    def setUp(self):
        """Initialize the unit test."""
        times  = np.asfortranarray([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        values = np.asfortranarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        self.ansvals = np.asfortranarray(np.arange(values[0],values[-1]+0.1,1.0))
        self.dt = 0.5
        self.current_time = 0.5
        self.ts = InputTS(times, values)

    def tearDown(self):
        """Finalize the unit test."""
        del self.ts

    def test__getTimeInterval(self):
        """Test _getTimeInterval method."""

        self.assertEqual(self.ts._getTimeInterval(0.0), 0)
        self.assertEqual(self.ts._getTimeInterval(0.5), 0)
        self.assertEqual(self.ts._getTimeInterval(1.0), 0)
        self.assertEqual(self.ts._getTimeInterval(2.0), 1)
        self.assertEqual(self.ts._getTimeInterval(3.0), 2)
        self.assertEqual(self.ts._getTimeInterval(4.0), 2)
        self.assertEqual(self.ts._getTimeInterval(5.0), 3)
        self.assertEqual(self.ts._getTimeInterval(16.0), 4)
        self.assertEqual(self.ts._getTimeInterval(1.000000000000001), 1)
        self.assertEqual(self.ts._getTimeInterval(2.000000000000001), 2)
        self.assertEqual(self.ts._getTimeInterval(4.000000000000001), 3)
        self.assertEqual(self.ts._getTimeInterval(8.000000000000001), 4)

    def test_updateTimeInterval(self):
        """Test updateTimeInterval method."""

        time = 1.9999999999
        self.ts.updateTimeInterval(time)
        self.assertEqual(self.ts.current_index, 1)

        time = 2.0
        self.ts.updateTimeInterval(time)
        self.assertEqual(self.ts.current_index, 1)

        time = 2.0000000001
        self.ts.updateTimeInterval(time)
        self.assertEqual(self.ts.current_index, 2)

        time = 8.000000000001
        self.ts.updateTimeInterval(time)
        self.assertEqual(self.ts.current_index, 4)

        time = 16.0
        self.ts.updateTimeInterval(time)
        self.assertEqual(self.ts.current_index, 4)

    def test_main(self):
        """Run the unit test."""

        self.ts.current_index = self.ts._getTimeInterval(self.current_time)
        self.assertEqual(self.ts.current_index, 0)

        expected_index = 0
        ansIndex = 0
        while (self.current_time <= self.ts.times[-1]):
            self.ts.updateTimeInterval(self.current_time)
            self.value = self.ts.interpolate(self.current_time)

            if (LOCALDEBUG==1 and pa.utilities.debug==pa.utilities.on):
                print("Current time = {0}".format(self.current_time))
                print("    Current value = {0}, expected value = {1}".format(
                        self.value, self.ansvals[ansIndex]))
                print("    Current index = {0}, expected index = {1}".format(
                        self.ts.current_index, expected_index))

            self.assertEqual(self.ts.current_index, expected_index)
            self.assertEqual(self.value, self.ansvals[ansIndex])

            self.current_time += self.dt

            ansIndex += 1
            if (self.current_time>self.ts.times[self.ts.current_index+1]):
                expected_index+=1

        return None

#------------------------------------------------------------------------------#
if __name__ == '__main__':
    unittest.main()

