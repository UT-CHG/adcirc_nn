#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#
import numpy as np

################################################################################
DEBUG_LOCAL = 1

################################################################################
def adcirc_init_bc_from_nn_hydrograph(anns): # anns is an AdcircNN_class object

    from .adcirc_nn_class import SERIESLENGTH, TIME_TOL

    ######################################################
    #SET UP ADCIRC BC series and edgestring.
    ######################################################
    assert(anns.pb.ibtype[anns.adcircedgestringid] == 22)

    ######################################################
    # Store the length of the coupled ADCIRC edge string
    for inode in range(1,anns.adcircedgestringnnodes):
        # -1 needed below since Python is 0 indexed whereas Fortan node numbers are 1-indexed
        n1 = anns.pb.nbvv[anns.adcircedgestringid][inode  ]-1
        n2 = anns.pb.nbvv[anns.adcircedgestringid][inode+1]-1
        x1 = anns.pm.x[n1]
        y1 = anns.pm.y[n1]
        x2 = anns.pm.x[n2]
        y2 = anns.pm.y[n2]
        delx = x2-x1
        dely = y2-y1
        anns.adcircedgestringlen += np.sqrt(delx*delx+dely*dely)
        if anns.pu.debug == anns.pu.on and DEBUG_LOCAL != 0:
            print(f"Nodes {n1}({x1},{y1}),{n2}({x2},{y2})")
    if anns.pu.messg==anns.pu.on:
        anns.adcircedgestringlen = anns.pmsg.pymessg_dbl_sum(anns.adcircedgestringlen, anns.adcirc_comm_comp)
    if anns.pu.debug == anns.pu.on and DEBUG_LOCAL != 0:
        print(f"Edge string({anns.adcircedgestringid+1}): Length = {anns.adcircedgestringlen}")

    ######################################################
    # Find series to modify during coupling.
    assert (SERIESLENGTH>=2)

    if anns.pu.debug == anns.pu.on and DEBUG_LOCAL != 0 and anns.myid==0:
        nbvStartIndex=sum(anns.pb.nvell[:anns.adcircedgestringid])
        print(f"Original: Flux time increment FTIMINC = {anns.pg.ftiminc}"
                f"\nOriginal: Flux times:\nQTIME1 = {anns.pg.qtime1}"
                f"\nQTIME2 = {anns.pg.qtime2}"
                f"\nOriginal: Flux values:\nQNIN1  = {anns.pg.qnin1[nbvStartIndex : nbvStartIndex+anns.adcircedgestringnnodes]}"
                f"\nQNIN2  = {anns.pg.qnin2[nbvStartIndex : nbvStartIndex+anns.adcircedgestringnnodes]}")

    ##################################################
    # Replace the flux time increment value.
    if anns.adcirctstart > 0.0: # NN starting time is before ADCIRC starting time; NN always starts at 0.0, hopefully!
        superdt = anns.adcircdt
    else:
        superdt = anns.adcirctstart
        while (superdt < anns.effectivenndt - TIME_TOL):
            superdt += anns.adcircdt #max(anns.effectivenndt, anns.adcircdt)
        superdt -= anns.adcirctstart

    #assert(anns.adcircdt>anns.nn.dt) #If this is true, then superdt = either adcirctstart or adcircdt.
    anns.pg.ftiminc = superdt

    ######################################################
    # Close the original fort.20, write a new one with a different name, and reopen it for reading
    errorio = anns.pu.pycloseopenedfileforread(20)
    assert(errorio==0)

    # Replace the fort.20 file.
    with open(anns.adcircfort20pathname, 'w') as fort20file:
        [fort20file.write('0.0\n') for i in range(anns.adcircedgestringnnodes*SERIESLENGTH)]

    errorio = anns.pg.pyopenfileforread(20,anns.adcircfort20pathname)
    assert(errorio==0)

    ##################################################
    # Replace the flux times and values.
    nbvStartIndex=sum(anns.pb.nvell[:anns.adcircedgestringid])
    anns.pg.qnin2[nbvStartIndex : nbvStartIndex+anns.adcircedgestringnnodes+1] = 0.0
    with open(anns.adcircfort20pathname, 'w') as fort20file:
        #Set series value to zero
        #anns.adcircseries[0].entry[i].value[0] = 0.0
        for dumm in range(SERIESLENGTH):
            for i in range(anns.pb.nvel):
                if anns.pb.lbcodei[i] in [2, 12, 22]:
                    [fort20file.write('{0:10f}\n'.format(anns.pg.qnin2[i]))]
                if anns.pb.lbcodei[i] == 32:
                    [fort20file.write('{0:10f}  {1:10f}\n'.format(anns.pg.qnin2[i],anns.pg.enin2[i]))]
        #Set starting time to <whatever>
        #anns.adcircseries[0].entry[i].time = anns.adcirctstart + i*superdt
        #if anns.couplingtype == 'AdndA':
        #    anns.adcircseries[0].entry[i].time += superdt # If 2-way AdndA, then time series has to be shifted ahead since ADCIRC goes first.
    anns.pg.qtime1 = anns.adcirctstart-superdt
    anns.pg.qtime2 = anns.pg.qtime1+anns.pg.ftiminc
    if anns.couplingtype == 'AdndA':
        anns.pg.qtime1 += superdt
        anns.pg.qtime2 += superdt
    #for i in range(anns.adcircseries[0].size):
    #    anns.adcircseries[0].entry[i].value[0] = 0.0
    #    anns.adcircseries[0].entry[i].time = anns.adcirctstart - (anns.adcircseries[0].size-2-i)*superdt
    #    if anns.couplingtype == 'AdndA':
    #        anns.adcircseries[0].entry[i].time += superdt # If 2-way AdndA, then time series has to be shifted ahead since ADCIRC goes first.

    # Play around with this if you are having problem with ADCIRC BC Series Starting time.
    if anns.adcirctstart > 0:  # If ADCIRC starting time is later than NN. NN always starts at 0.
        #for i in range(SERIESLENGTH-2):
        #    anns.adcircseries[0].entry[i].time -= anns.adcirctstart
        if anns.couplingtype != 'AdndA':
            #anns.adcircseries[0].entry[anns.adcircseries[0].size-2].time -= anns.adcirctstart
            anns.pg.qtime1 -= anns.adcirctstart
            anns.pg.ftiminc += anns.adcirctstart ## Gajanan gkc warning caution: Newly added in 03/2020
                                               ## Ensure this gets replaced in set_bc function.

    if anns.pu.debug == anns.pu.on and DEBUG_LOCAL != 0:
        print(f"Replaced: Flux time increment FTIMINC = {anns.pg.ftiminc} "
                f"\nReplaced: Flux times:\nQTIME1 = {anns.pg.qtime1} "
                f"\nQTIME2 = {anns.pg.qtime2} "
                f"\nReplaced: Flux values:\nQNIN1  = {anns.pg.qnin1[nbvStartIndex : nbvStartIndex+anns.adcircedgestringnnodes]} "
                f"\nQNIN2  = {anns.pg.qnin2[nbvStartIndex : nbvStartIndex+anns.adcircedgestringnnodes]}")

################################################################################
if __name__ == '__main__':
    pass

