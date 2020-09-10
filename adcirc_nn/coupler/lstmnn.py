#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
The Long Short Term Memory Neural Network module.
"""
import numpy as np

NN_TIME_FACTOR = 1.0 # No conversion for now. Seconds to seconds

#------------------------------------------------------------------------------#
class LongShortTermMemoryNN_class():
    """The Long Short Term Memory Neural Network class."""

    #--------------------------------------------------------------------------#
    def __init__(self):
        """Construct LSTM NN object."""

        self._DEBUG = 0

        self.btime = 0.0 # Double in Julian date
        self.dt = 0.0  # Double in seconds
        self.timer = 0 # Integer in seconds
        self.niter = 0 # Integer in seconds
        self.single_event_end = 0 # Integer in minutes
        self.go = 1 # Integer flag for running or not running the model

        self.tprev = 0.0 # Double in seconds  # To be set to timer
        self.tfinal = 0.0 # Double in seconds # To be set to niter
        self.voutprev=0.0
        self.voutprev_t=0.0

        self.timefact=NN_TIME_FACTOR # Minutes to seconds conversion, since niter is in mins.

        #self.dummytimes = 0.0
        #self.dummyvalues = 0.0
        self.vout = 0.0
        self.qout = 0.0

        self.runflag = 1 # Only for use in coupling with ADCIRC.

    #--------------------------------------------------------------------------#
    def initialize(self):
        """Initialize LSTM NN object."""

        # Hard-coded values for now:
        self.dt = 60.0
        self.timer = 0
        self.niter = 21600
        self.qbcfunc = lambda t : 1.0e1*(1-np.cos(2.0*np.pi * t / self.tfinal))
        self.vbcfunc = lambda t : 1.0e1*(t-np.sin(2.0*np.pi * t / self.tfinal)*self.tfinal/4.0/np.pi)
        #self.dummytimes = np.arange(0.0, self.dt*5, self.tfinal)
        #self.dummyvalues = 1.0e3*(1-np.cos(4.0*np.pi/self.dummytimes))

        self.btime = 0.0
        self.tprev = self.timer
        self.tfinal = self.niter

    #--------------------------------------------------------------------------#
    def run(self):
        """Run the LSTM NN object."""

        # Run the NN model. For now, just set the dummy value for next "t"
        while (self.timer < self.niter):
            self.vout = self.vbcfunc(self.timer + self.dt)
            self.qout = self.qbcfunc(self.timer + self.dt)
            # Increment model time
            self.timer += self.dt

        return 0

    #--------------------------------------------------------------------------#
    def finalize(self):
        """Finalize LSTM NN object."""
        # Do nothing for now
        pass

