#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
The Long Short Term Memory Neural Network module.
"""

import os
from os.path import join as ospathjoin
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import yaml

from .RunoffLSTM import RunoffLSTM

NN_TIME_FACTOR = 3600.0 # No conversion for now. Seconds to seconds
NN_LENGTH_FACTOR = 0.3048 # Ft to meter conversion

#------------------------------------------------------------------------------#
class LongShortTermMemoryNN_class():
    """The Long Short Term Memory Neural Network class."""

    #--------------------------------------------------------------------------#
    def __init__(self):
        """Construct LSTM NN object."""

        self._DEBUG = 0

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
        self.length_factor = NN_LENGTH_FACTOR # TODO: remember to do this conversion from adc to nn

        #self.dummytimes = 0.0
        #self.dummyvalues = 0.0
        self.elev = 0.0

        self.runflag = 1 # Only for use in coupling with ADCIRC.

    #--------------------------------------------------------------------------#
    def initialize(self):
        """Initialize LSTM NN object."""

        # Hard-coded values for now:
        self.dt = 1 # 1 for one hour for now
        self.timer = 0
        self.niter = 100
        # self.elbcfunc = lambda t : 5.0e0*(1-np.cos(2.0*np.pi * t / self.tfinal))
        #self.dummytimes = np.arange(0.0, self.dt*5, self.tfinal)
        #self.dummyvalues = 1.0e3*(1-np.cos(4.0*np.pi/self.dummytimes))

        self.tprev = self.timer
        self.tfinal = self.niter

        # load LSTM model
        nn_input_dir = "nn_input"
        self.nn_model = self._load_nn_model(nn_input_dir) # provide path if not in the same folder
        self.hidden = (self.nn_model.c0, self.nn_model.h0)

        # load nn input data
        df = pd.read_csv(ospathjoin(nn_input_dir, "event.csv"))
        features = [
            '43057',
            '43060',
            '43053',
            '43059',
            '43047',
            '43058',
            '43052',
            '43050',
            '43160',
            '43054',
            '43051',
            '43049',
            '43056',
            '43091',
            '43055',
            "Verified (ft)",
        ]
        self.features = df[features] # TODO: overwrite this variable during two way coupling
        # separting rainfall and wse can be done by
        # self.rainfall = self.features[rain_gages]
        # self.wl = self.features["Verified (ft)"]
        # and resemble them after overwriting wl
        self.X_scaler, self.y_scaler = self._load_meta(nn_input_dir)
        assert (self.niter <= len(df)), "lstm model does not have enough input data"

    #--------------------------------------------------------------------------#
    def run(self):
        """
        Run the LSTM NN object.
        """
        while (self.timer < self.niter):
            # Time step of NN model
            X = self.X_scaler.transform(
                self.features.loc[self.timer].values.reshape(1, -1)
            )
            X = torch.Tensor(X).view(1, 1, -1)
            y, self.hidden = self.nn_model(X, self.hidden)
            y = y.view(-1).detach().numpy()
            self.elev = self.y_scaler.inverse_transform(y.reshape(1, -1)) * self.length_factor

            # Increment model time
            self.timer += self.dt

        return 0

    #--------------------------------------------------------------------------#
    def finalize(self):
        """Finalize LSTM NN object."""
        # Do nothing for now
        pass


    def _load_meta(self,path=''):
        with open(ospathjoin(path, "meta_data.pkl"), 'rb') as meta_data:
            X_scaler, y_scaler = pickle.load(meta_data)
        
        return X_scaler, y_scaler


    def _load_nn_model(self, path=''):
        # load model parameters
        with open(ospathjoin(path, "model_parameters.yaml"), 'r+') as f:
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
            ospathjoin(path,"model.pt"),
            map_location=lambda storage,
            location: storage
        ) # load cuda model to cpu machine
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Number of epoch: {epoch}".format(epoch=epoch))
        print("Training loss: {loss}".format(loss=loss)) # TODO: better message

        model.eval()

        return model

