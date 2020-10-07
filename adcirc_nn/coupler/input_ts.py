#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
Module for the InputTS class for storing Neural Network's time series.
"""

#------------------------------------------------------------------------------#
class InputTS():
    """The InputTS class for storing Neural Network's time series."""

    #--------------------------------------------------------------------------#
    def __init__(self, times, values):
        '''InputTS class constructor.'''

        self.times = times
        self.values = values
        self.current_index = 0

    #--------------------------------------------------------------------------#
    def __del__(self):
        '''InputTS class constructor.'''

        del self.times, self.values, self.current_index

    #--------------------------------------------------------------------------#
    def _getTimeInterval(self, t):
        '''Get the time interval of the current time stamp.

           Only used when initializing the object.
           Returns index such that t lies in the interval,
           (  times[index],  times[index+1]  ]
           Uses binary search to get index since the time
           stamps are sorted.
        '''

        ntimes = len(self.times)
        mid = 0
        if (ntimes==0):
            self.current_index = -1
            return -1
        else:
            left = 0
            while (ntimes!=1):
                mid=ntimes//2
                if (t<=self.times[left+mid]):
                    ntimes=mid
                else:
                    ntimes-=mid
                    left+=mid
            self.current_index = left
            return left

    #--------------------------------------------------------------------------#
    def updateTimeInterval(self, t):
        '''Update the time interval by updating current_index.

           Returns index such that the time, t, lies in the interval,
           (  times[index],  times[index+1]  ]
           If t is greater than the last time stamp, an out-of-bounds
           error is thrown.
           If t is less than the first time stamp, then index=0 is
           returned. In this case, the value gets extrapolated behind
           using the first time interval in the series.
        '''

        while (self.times[self.current_index+1] < t):
            self.current_index+=1

    #--------------------------------------------------------------------------#
    def interpolate(self, time):
        '''Interpolate input series onto NN model's current time.'''

        slope = (self.values[self.current_index+1] \
                  - self.values[self.current_index]) \
              / (self.times[self.current_index+1] \
                  - self.times[self.current_index])
        value = self.values[self.current_index] \
              + slope * (time-self.times[self.current_index])

        return value

