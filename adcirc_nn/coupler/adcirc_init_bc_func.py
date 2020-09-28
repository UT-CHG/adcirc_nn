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
    # Find series to modify during coupling.
    assert (SERIESLENGTH>=2)

    if anns.pu.debug == anns.pu.on and DEBUG_LOCAL != 0 and anns.myid==0:
        db_el_StartIndex=sum(anns.pb.nvdll[:anns.adcircedgestringid])
        print(f"Original: Elevation series time increment ETIMINC = {anns.pg.etiminc}"
                f"\nOriginal: Elev times:\nETIME1 = {anns.pg.etime1}"
                f"\nETIME2 = {anns.pg.etime2}"
                f"\nOriginal: Elev values:\nESBIN1  = {anns.pg.esbin1[db_el_StartIndex : db_el_StartIndex+anns.adcircedgestringnnodes]}"
                f"\nESBIN2  = {anns.pg.esbin2[db_el_StartIndex : db_el_StartIndex+anns.adcircedgestringnnodes]}")

    ##################################################
    # Replace the elevation time increment value.
    if anns.adcirctstart > 0.0: # NN starting time is before ADCIRC starting time; NN always starts at 0.0, hopefully!
        superdt = anns.adcircdt
    else:
        superdt = anns.adcirctstart
        while (superdt < anns.effectivenndt - TIME_TOL):
            superdt += anns.adcircdt #max(anns.effectivenndt, anns.adcircdt)
        superdt -= anns.adcirctstart

    #assert(anns.adcircdt>anns.nn.dt) #If this is true, then superdt = either adcirctstart or adcircdt.
    anns.pg.etiminc = superdt

    ######################################################
    # Close the original fort.19, write a new one with a different name, and reopen it for reading
    errorio = anns.pu.pycloseopenedfileforread(19)
    assert(errorio==0)

    # Replace the fort.19 file.
    with open(anns.adcircfort19pathname, 'w') as fort19file:
        [fort19file.write('0.0\n') for i in range(anns.adcircedgestringnnodes*SERIESLENGTH)]

    errorio = anns.pg.pyopenfileforread(19,anns.adcircfort19pathname)
    assert(errorio==0)

    ##################################################
    # Replace the elevation times and values.
    db_el_StartIndex=sum(anns.pb.nvdll[:anns.adcircedgestringid])
    anns.pg.esbin2[db_el_StartIndex : db_el_StartIndex+anns.adcircedgestringnnodes+1] = 0.0
    with open(anns.adcircfort19pathname, 'w') as fort19file:
        #Set series value to zero
        #anns.adcircseries[0].entry[i].value[0] = 0.0
        for dumm in range(SERIESLENGTH):
            for i in range(anns.pb.neta):
                [fort19file.write('{0:10f}\n'.format(anns.pg.esbin2[i]))]
        #Set starting time to <whatever>
        #anns.adcircseries[0].entry[i].time = anns.adcirctstart + i*superdt
        #if anns.couplingtype == 'AdndA':
        #    anns.adcircseries[0].entry[i].time += superdt # If 2-way AdndA, then time series has to be shifted ahead since ADCIRC goes first.
    anns.pg.etime1 = anns.adcirctstart-superdt
    anns.pg.etime2 = anns.pg.etime1+anns.pg.etiminc
    if anns.couplingtype == 'AdndA':
        anns.pg.etime1 += superdt
        anns.pg.etime2 += superdt
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
            anns.pg.etime1 -= anns.adcirctstart
            anns.pg.etiminc += anns.adcirctstart ## Gajanan gkc warning caution: Newly added in 03/2020
                                               ## Ensure this gets replaced in set_bc function.

    if anns.pu.debug == anns.pu.on and DEBUG_LOCAL != 0:
        print(f"Replaced: Elev time increment ETIMINC = {anns.pg.etiminc} "
                f"\nReplaced: Elev times:\nETIME1 = {anns.pg.etime1} "
                f"\nETIME2 = {anns.pg.etime2} "
                f"\nReplaced: Elev values:\nESBIN1  = {anns.pg.esbin1[db_el_StartIndex : db_el_StartIndex+anns.adcircedgestringnnodes]} "
                f"\nESBIN2  = {anns.pg.esbin2[db_el_StartIndex : db_el_StartIndex+anns.adcircedgestringnnodes]}")

################################################################################
if __name__ == '__main__':
    pass

