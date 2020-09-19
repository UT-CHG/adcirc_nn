#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
The Long Short Term Memory Neural Network module.
"""
import numpy as np
import yaml
import torch

from RunoffLSTM import RunoffLSTM

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
        self.elevprev=0.0
        self.elevprev_t=0.0

        self.timefact=NN_TIME_FACTOR # Minutes to seconds conversion, since niter is in mins.

        #self.dummytimes = 0.0
        #self.dummyvalues = 0.0
        self.elev = 0.0

        self.runflag = 1 # Only for use in coupling with ADCIRC.

    #--------------------------------------------------------------------------#
    def initialize(self):
        """Initialize LSTM NN object."""

        # Hard-coded values for now:
        self.dt = 60.0
        self.timer = 0
        self.niter = 21600
        self.elbcfunc = lambda t : 5.0e0*(1-np.cos(2.0*np.pi * t / self.tfinal))
        #self.dummytimes = np.arange(0.0, self.dt*5, self.tfinal)
        #self.dummyvalues = 1.0e3*(1-np.cos(4.0*np.pi/self.dummytimes))

        self.btime = 0.0
        self.tprev = self.timer
        self.tfinal = self.niter

        # load LSTM model
        self.nn_model = self._load_nn_model() # provide path if not in the same folder


    #--------------------------------------------------------------------------#
    def run(self):
        """Run the LSTM NN object."""

        # Run the NN model. For now, just set the dummy value for next "t"
        while (self.timer < self.niter):
            self.elev = self.elbcfunc(self.timer + self.dt)
            # Increment model time
            self.timer += self.dt

            # Time step of NN model
            if first_step: # TODO: self.timer == 0?
                hidden = (self.nn_model.c0, self.nn_model.h0)
            y, hidden = self.nn_model(X, hidden) # TODO: X is input to the nn model

        return 0

    #--------------------------------------------------------------------------#
    def finalize(self):
        """Finalize LSTM NN object."""
        # Do nothing for now
        pass

    def _load_nn_model(self, path=''):
        # load model parameters
        with open(path+"model_parameters.yaml", 'r+') as f:
            model_parameters = yaml.load(f)
        features = model_parameters['features']
        input_dim = len(features)
        hidden_size = model_parameters['hidden_size']
        output_dim = model_parameters['output_dim']
        num_layers = model_parameters['num_layers']
        learning_rate = model_parameters['learning_rate']
        activation = model_parameters['activation']
        dropout = model_parameters['dropout']
        l2_regularization = model_parameters['l2_regularization']
        
        # instantiate nn and optimizer
        model = RunoffLSTM(
            input_dim, 
            hidden_size, 
            output_dim, 
            num_layers, 
            dropout,
            activation,
        )

        # instantiate optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            amsgrad=True, 
            weight_decay=l2_regularization
        )

        # load PyTorch checkpoint
        checkpoint = torch.load(
            path+"model.pt", 
            map_location=lambda storage, 
            location: storage
        ) # load cuda model to cpu machine
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Number of epoch: {epoch}".format(epoch=epoch))
        print("Training loss: {loss}".format(epoch=loss)) # TODO: better message

        model.eval()

        return model

