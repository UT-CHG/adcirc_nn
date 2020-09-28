#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
"""
The main adcirc-nn class.
"""
import numpy as np

from pyADCIRC import libadcpy

from .adcirc_init_bc_func import adcirc_init_bc_from_nn_hydrograph
from .adcirc_set_bc_func  import adcirc_set_bc_from_nn_hydrograph
from .lstmnn import LongShortTermMemoryNN_class as nn

#------------------------------------------------------------------------------#
TIME_TOL = 1.0e-3
SERIESLENGTH = 4 #This is the MINIMUM number of lines required in an ADCIRC series to be coupled. Compulsory.
DEBUG_LOCAL = 1

#------------------------------------------------------------------------------#
class AdcircNN():
    """The main class of adcirc-nn."""

    #--------------------------------------------------------------------------#
    def __init__(self):
        """Inititialize AdcircNN class."""
        self.pa    = libadcpy
        self.ps    = libadcpy.sizes
        self.pg    = libadcpy.pyglobal
        self.pm    = libadcpy.pymesh
        self.pmsg  = libadcpy.pymessenger
        self.pb    = libadcpy.pyboundaries
        self.pmain = libadcpy.pyadcirc_mod
        self.pu    = libadcpy.utilities

        self.couplingtype = ""
        self.couplingntsteps = 1 # No. of ADCIRC time steps between coupled intervals; minimum = 1
        self.couplingdtfactor = 1
        self.npes = 0
        self.myid = 0

        # ADCIRC data
        self.adcircrunflag=self.pu.on
        self.adcircdt=0.0
        self.adcircnt=0
        self.adcirctstart=0.0
        self.adcirctprev=0.0
        self.adcirctnext=0.0
        self.adcirctfinal=0.0
        self.adcircntsteps=0
        self.adcirc_comm_world=0
        self.adcirc_comm_comp=0

        self.adcircseries=0
        self.adcircedgestringid=self.pu.unset_int
        self.adcircedgestringnnodes=self.pu.unset_int
        self.adcircedgestringlen=0.0
        self.adcircfort19pathname=''
        self.adcirc_hprev=0.0   # Avg depth
        self.adcirc_hprev_len=0.0   # count

        # Neural Network data
        self.nn = nn()
        self.effectivenndt=0.0

    #--------------------------------------------------------------------------#
    def coupler_initialize(self, argc, argv):
        """Initialize the ADCIRC model and the Neural Network."""

        # argv[argc-2] must be coupling type from {Adn, ndA, AdndA, ndAdn}
        # argv[argc-1] must be edge string ID of the ADCIRC model
        ######################################################
        #SET UP ADCIRC.
        ######################################################
        if self.pu.debug == self.pu.on and DEBUG_LOCAL != 0:
            print("\nInitializing ADCIRC\n")
        self.pmain.pyadcirc_init()
        self.npes = self.ps.mnproc
        self.myid = self.ps.myproc
        if self.pu.messg == self.pu.on:
            self.adcirc_comm_world = self.pmsgmpi_comm_adcirc
            self.adcirc_comm_comp = self.pgcomm
        if self.pu.debug == self.pu.on and DEBUG_LOCAL != 0:
            print("MPI Info: npes =", self.npes, ", myid =", self.myid)

        ######################################################
        if self.pu.messg == self.pu.on:
            #self.pmsgmsg_init()
            if (self.pu.debug == self.pu.on and DEBUG_LOCAL!=0):
                print('Python: adcirc_comm_world pointer value : '+hex(self.adcirc_comm_world))
                print('Python: adcirc_comm_comp  pointer value : '+hex(self.adcirc_comm_comp))
            print("*********************** MPI Initialized ***********************")
            print("***************************************************************")


        print("********************* ADCIRC Initialized **********************")
        print("***************************************************************")

        self.couplingtype=argv[argc.value-2]
        #self.couplingdtfactor = 480
        self.adcircrunflag=self.pu.on
        self.adcirctstart=0.+self.pg.statim*86400.0 #statim is in days.
        self.adcircdt=0.+self.pg.dt*float(self.couplingdtfactor) #Needed 0+ to prevent the two from being the same object :-/ Careful!!!!
        self.adcircnt=0+self.pg.nt #Needed 0+ to prevent the two from being the same object :-/ Careful!!!!
        self.adcirctprev=self.adcirctstart
        self.adcirctnext=self.adcirctprev
        self.adcirctfinal=(self.pg.statim + self.pg.rnday)*86400.0
        self.adcircntsteps=0+self.pmain.itime_end #Needed 0+ to prevent the two from being the same object :-/ Careful!!!!
        self.adcircfort19pathname=''.join(np.append(np.char.strip(self.ps.inputdir),'/fort.19.new'))
        self.adcircedgestringid=int(argv[argc.value-1])-1
        self.adcircedgestringnnodes=self.pb.nvell[self.adcircedgestringid]
        self.adcircedgestringnodes=self.pb.nbvv[1:self.adcircedgestringnnodes]

        self.nn.initialize()
        self.nn.runflag=self.pu.on
        self.nn._DEBUG=self.pu.on
        self.effectivenndt=self.nn.dt*self.nn.timefact # MUST be in seconds. This is in case we decide to use single_event_end time as ending time
        self.nntprev=self.nn.timer
        self.nntfinal=self.nn.niter



    #--------------------------------------------------------------------------#
    def coupler_finalize(self):
        """Finalize the ADCIRC model and the neural network."""
        if self.pu.debug==self.pu.on and DEBUG_LOCAL!=0 and self.myid==0:
            print('\n\nFinalizing ADCIRC\n')
        ierr_code = self.pmain.pyadcirc_finalize()
        if self.myid==0:
            print("********************** ADCIRC Finalized ***********************")
            print("***************************************************************")

    #--------------------------------------------------------------------------#
    def coupler_run_nn_driving_adcirc(self):
        """Run function with NN staying ahead of ADCIRC."""

        # Todo
        adcirc_init_bc_from_nn_hydrograph(self)
        if self.couplingtype == 'ndAdn':
            nn_init_bc_from_adcirc_depths(self)

        # Set final times to zero.
        self.pmain.itime_end = 0
        self.nn.niter = 0
        # Run NN only on 1 processsor: PE 0.
        if self.myid == 0:
            ierr_code = self.nn.run()
            assert(ierr_code == 0)
            self.nn.go    = self.pu.on
        else:
            # Assumes NN cannot start at negative time!
            self.nn.go    = self.pu.off
        if self.pu.messg == self.pu.on:
            if (self.pu.debug ==self.pu.on or DEBUG_LOCAL != 0):
                print('PE[{}] Before messg: timer = {}'.format(self.myid,self.nn.timer))
            self.nn.timer = self.pmsg.pymsg_dbl_max(self.nn.timer, self.adcirc_comm_comp)
            if (self.pu.debug ==self.pu.on or DEBUG_LOCAL != 0):
                print('PE[{}] After messg : timer = {}'.format(self.myid,self.nn.timer))

        while (self.adcirctprev<self.adcirctfinal or self.nn.timer<self.nntfinal):
            ######################################################
            if (self.nn.timer < self.nntfinal):
                # Decided while writing report. Driving model must take at least one time step forward.
                superdt = self.effectivenndt
                while (self.nn.timer*self.nn.timefact + superdt < self.adcirctprev+self.adcircdt-TIME_TOL):
                    superdt                    += self.effectivenndt

                self.nn.niter               += int(max(1.0, (superdt+TIME_TOL)/self.nn.timefact))
                # This one is the important one that determines end time:
                #self.nn.single_event_end     = self.nn.b_lt_start + (self.nn.timer*self.nn.timefact + superdt)/86400.0 #Julian

                if (self.adcircrunflag==self.pu.off): #If ADCIRC is done first, let NN finish off directly.
                    self.nn.niter            = self.nntfinal
                    #self.nn.single_event_end = self.nn.b_lt_start + self.nn.niter/1440.0 #nntfinal was original niter in mins

                if self.nn._DEBUG == self.pu.on and DEBUG_LOCAL != 0 and self.myid == 0:
                    print("\n*******************************************\nRunning NN:")
                    print("dt             =", self.nn.dt)
                    print("timer          =", self.nn.timer)
                    print("niter          =", self.nn.niter)
                    print("superdt        =", superdt)
                    print("end time       =", self.nn.timer*self.nn.timefact + superdt)
                elif self.myid==0:
                    print("\n*******************************************\nRunning NN:")

                # Run NN only on 1 processsor: PE 0.
                if self.myid == 0:
                    ierr_code = self.nn.run()
                    assert(ierr_code == 0)
                    # Needed to force nn to run for next time step:
                    self.nn.go    = self.pu.on
                else:
                    # Note: We are keeping nn.runflag as on, but nn.go as FALSE!!
                    # This matters in adcirc_set_bc functions!
                    self.nn.go    = self.pu.off
                if self.pu.messg == self.pu.on:
                    if (self.pu.debug ==self.pu.on or DEBUG_LOCAL != 0):
                        print('PE[{}] Before messg: timer = {}'.format(self.myid,self.nn.timer))
                    self.nn.timer = self.pmsg.pymsg_dbl_max(ctypes_c_double(self.nn.timer), self.adcirc_comm_comp)
                    if (self.pu.debug ==self.pu.on or DEBUG_LOCAL != 0):
                        print('PE[{}] After messg : timer = {}'.format(self.myid,self.nn.timer))

            else:
                self.nn.runflag = self.pu.off
                self.nn.go    = self.pu.off

            ######################################################
            # Set ADCIRC Boundary conditions from NN
            adcirc_set_bc_from_nn_hydrograph(self)

            ######################################################
            if (self.adcirctprev < self.adcirctfinal):
                ntsteps = 0
                while (self.adcirctnext < self.nn.timer*self.nn.timefact-self.adcircdt+TIME_TOL):
                    ntsteps += self.couplingdtfactor
                    self.adcirctnext += self.adcircdt
                if (self.nn.runflag == self.pu.off):
                    ntsteps = (self.adcircntsteps-self.pmain.itime_bgn+1)
                    self.adcirctnext = self.adcirctfinal

                if self.pu.debug == self.pu.on and DEBUG_LOCAL != 0 and self.myid == 0:
                    print("\n****************************************\nRunning ADCIRC:")
                    print("dt             =", self.adcircdt)
                    print("t_prev         =", self.adcirctprev)
                    print("t_final        =", self.adcirctnext)
                    print("ntsteps        =", ntsteps)
                elif self.myid==0:
                    print("\n****************************************\nRunning ADCIRC:")

                # Run ADCIRC
                self.pmain.pyadcirc_run(ntsteps)
                self.adcirctprev = (self.pmain.itime_bgn-1)*self.pg.dtdp + self.pg.statim*86400.0

            else:
                self.adcircrunflag=self.pu.off

            ######################################################
            ## Set NN Boundary conditions from ADCIRC
            if self.couplingtype == 'ndAdn':
                nn_set_bc_from_adcirc_depths(self)

    #--------------------------------------------------------------------------#
    def coupler_run(self):
        """Run the physics based machine learning ADCIRC model."""

        print("\n\n***************************************************************")
        print(    "***************************************************************")
        if self.couplingtype == 'ndA':
            run_string = 'Running NN driving ADCIRC, One-way coupling'
            print(run_string)
            self.coupler_run_nn_driving_adcirc()

        elif self.couplingtype == 'Adn':
            run_string = 'Running ADCIRC driving NN, One-way coupling'
            print(run_string)
            self.coupler_run_adcirc_driving_nn()

        elif self.couplingtype == 'ndAdn':
            run_string = 'Running NN driving ADCIRC driving NN, Two-way coupling'
            print(run_string)
            self.coupler_run_nn_driving_adcirc()

        elif self.couplingtype == 'ndAdn':
            run_string = 'Running ADCIRC driving NN driving ADCIRC, Two-way coupling'
            print(run_string)
            self.coupler_run_adcirc_driving_nn()

        else:
            print('Unkown coupling type supplied by user: {}\nExiting.'.format(self.couplingtype))
            return

        print("\n\n***************************************************************")
        print(    "***************************************************************")
        print(    "Finished", run_string)
        print(    "***************************************************************")
        print(    "***************************************************************")

