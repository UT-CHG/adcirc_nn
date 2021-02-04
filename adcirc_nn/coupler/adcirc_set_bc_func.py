#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#

############################################################################################################################################################
DEBUG_LOCAL = 1
############################################################################################################
def adcirc_set_bc_from_nn_hydrograph(ags): # ags is an Adcirc_NN_class object.

    from .adcirc_nn_class import SERIESLENGTH, TIME_TOL

    ########## Set ADCIRC Boundary Conditions ###########
    # Note: ags.adcircseries already points to the head of series in ADCIRC that needs to be modified.
    db_el_StartIndex=sum(ags.pb.nvdll[:ags.adcircedgestringid])
    if ags.pu.debug == ags.pu.on and DEBUG_LOCAL != 0 and ags.myid==0:
        print(f"\nOriginal: Elev time increment ETIMINC = {ags.pg.etiminc}"
                f"\nOriginal: Elev times:\nETIME1 = {ags.pg.etime1}"
                f"\nETIME2 = {ags.pg.etime2}"
                f"\nOriginal: Elev values:\nESBIN1  = {ags.pg.esbin1[db_el_StartIndex : db_el_StartIndex+ags.adcircedgestringnnodes]}"
                f"\nESBIN2  = {ags.pg.esbin2[db_el_StartIndex : db_el_StartIndex+ags.adcircedgestringnnodes]}")

    if ags.pu.messg == ags.pu.on:
        if ags.myid != 0:
            messgelev  = -1.0E+200
        else:
            messgelev  = ags.nn.elev
        if (ags.pu.debug ==ags.pu.on or ags.nn._DEBUG == ags.pu.on) or DEBUG_LOCAL != 0:
            print(f'PE[{ags.myid}] Before messg: elev = {ags.nn.elev}')
        ags.nn.elev  = ags.pmsg.pymessg_dbl_max(messgelev, ags.adcirc_comm_comp)
        if (ags.pu.debug ==ags.pu.on or ags.nn._DEBUG == ags.pu.on) or DEBUG_LOCAL != 0:
            print(f'PE[{ags.myid}] After messg : elev = {ags.nn.elev}')

    ######################################################
    # Close the original fort.19
    errorio = ags.pu.pycloseopenedfileforread(19)
    assert(errorio==0)

    if (ags.nn.runflag != ags.pu.off):

        # Inflow volume in the current nn time step:
        DH = (ags.nn.elev-ags.nn.elevprev)
        if ags.couplingtype == 'AdndA':
            DT = 0.0
            ags.adcirctprev=ags.pu.pyfindelapsedtime(ags.pmain.itime_end) #Last time at which ADCIRC was paused & solution known
            while (ags.adcirctprev + DT < ags.nn.timer*ags.nn.timefact+ags.effectivenndt-TIME_TOL):
                DT += ags.adcircdt
        else: # For ndA and ndAdg:
            DT = (ags.nn.timer-ags.nn.elevprev_t)*ags.nn.timefact + 1.0E-20

        if (ags.pu.debug ==ags.pu.on or ags.nn._DEBUG == ags.pu.on) and DEBUG_LOCAL != 0 and ags.myid == 0:
            # This is valid only for PE 0 which is running NN. Not on other PEs!
            #outlet_area  = ags.nn.area[      ags.nn.nx[ags.nn.nlinks]][ags.nn.nlinks]
            #outlet_depth = ags.nn.chan_depth[ags.nn.nx[ags.nn.nlinks]][ags.nn.nlinks]
            #print(f"outlet area       = {outlet_area} m2")
            #print(f"outlet chan_depth = {outlet_depth} m")
            print(f"elevprev          = {ags.nn.elevprev} m")
            print(f"elev              = {ags.nn.elev} m")
            print(f"DH                = {DH} m")
            print(f"DT                = {DT} s")

        # Move current to previous: Current is at [2], previous is at [1]
        # Shift values backward
        ags.pg.etime1 = ags.pg.etime2
        for i in range(len(ags.pg.esbin1)):
            ags.pg.esbin1[i] = ags.pg.esbin2[i]

        # Set ADCIRC series value for nn time t2
        ags.pg.etime2 = ags.nn.timer*ags.nn.timefact # This is NN time set in ADCIRC series.
        if ags.couplingtype == 'AdndA':
            ags.pg.etime2 = ags.adcirctprev+DT #ags.adcircdt # If 2-way AdndA, then time series has to be shifted ahead since ADCIRC goes first.

        # ADCIRC Series value
        #DT_calculated = ags.adcircseries[0].entry[SERIESLENGTH-2].time - ags.adcircseries[0].entry[SERIESLENGTH-3].time
        #print"DT_calculated     =", DT_calculated, "s"
        #DT_calculated affects how the mass is distributed. If we want to dump all the mass from NN into ADCIRC's next time step
        #no matter how large it may be, we should use DT_calculated. For now, I'm skipping DT_calculated.
        oldseriesvalue = ags.pg.esbin2[db_el_StartIndex]
        #seriesvalue = (2*DH/DT/ags.adcircedgestringlen * ags.nn.hydrofact - oldseriesvalue)
        seriesvalue =  ags.nn.elev
        ags.pg.esbin2[db_el_StartIndex : db_el_StartIndex+ags.adcircedgestringnnodes] = seriesvalue
        with open(ags.adcircfort19pathname, 'w') as fort19file:
            #print "ESBIN values start at", db_el_StartIndex
            for i in range(ags.pb.neta):
                [fort19file.write('{0:10f}\n'.format(ags.pg.esbin2[i]))]
            # Now set the last value same as the current value, but not the time!
            # TO IMPLEMENT THIS PART, JUST WRITE THE SERIES TWICE IN fort.19 replacement!
            #ags.adcircseries[0].entry[SERIESLENGTH-1].time     = ags.adcircseries[0].entry[SERIESLENGTH-2].time + TIME_TOL
            #ags.adcircseries[0].entry[SERIESLENGTH-1].value[0] = ags.adcircseries[0].entry[SERIESLENGTH-2].value[0]
            for i in range(ags.pb.neta):
                [fort19file.write('{0:10f}\n'.format(ags.pg.esbin2[i]))]

        # Calculate slope
        ags.adcircseriesslope = \
                (seriesvalue - oldseriesvalue) / \
                (ags.pg.etime2 - ags.pg.etime1)#+1.0E-14)

        # Calculate 'area', i.e., volume/unit width that has flown in at this time step.
        ags.adcircseriesarea  = 0.5 * \
                (seriesvalue + oldseriesvalue) * \
                (ags.pg.etime2 - ags.pg.etime1)

        #Store volume for the next time step.
        ags.nn.elevprev   = ags.nn.elev
        ags.nn.elevprev_t = ags.nn.timer

    else:
        # Shift values backward
        ags.pg.etime1 = ags.pg.etime2
        for i in range(ags.pb.neta*2):
            ags.pg.esbin1[i] = ags.pg.esbin2[i]
        # Replace the fort.19 file.
        with open(ags.adcircfort19pathname, 'w') as fort19file:
            [fort19file.write('0.0\n') for i in range(ags.adcircedgestringnnodes*SERIESLENGTH)]
        # Reset the elevation time increment
        # Gajanan gkc warning: Note that this will cause a problem if there are multiple non-zero-flux boundaries!!!
        ags.pg.etiminc = abs(ags.adcirctfinal)*10.0

    ######################################################
    # Reopen the fort.19 replacement file
    errorio = ags.pg.pyopenfileforread(19,ags.adcircfort19pathname)
    assert(errorio==0)

    if ags.pu.debug == ags.pu.on and DEBUG_LOCAL != 0:
        print(f"Replaced: Elev time increment ETIMINC = {ags.pg.etiminc}"
                f"\nReplaced: Elev times:\nETIME1 = {ags.pg.etime1}"
                f"\nETIME2 = {ags.pg.etime2} "
                f"\nReplaced: Elev values:\nESBIN1  = {ags.pg.esbin1[db_el_StartIndex : db_el_StartIndex+ags.adcircedgestringnnodes]}"
                f"\nESBIN2  = {ags.pg.esbin2[db_el_StartIndex : db_el_StartIndex+ags.adcircedgestringnnodes]}")
        print(f'Area   contained  = {ags.adcircseriesarea}')
        print(f'Volume contained  = {ags.adcircseriesarea*ags.adcircedgestringlen}')


############################################################################################################
if __name__ == '__main__':
    pass
