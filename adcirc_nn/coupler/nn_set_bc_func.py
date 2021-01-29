#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#

############################################################################################################################################################
DEBUG_LOCAL = 1

############################################################################################################
def nn_set_bc_from_adcirc_depths(anns): # anns is of type adcircnntruct.

    from .adcirc_nn_class import SERIESLENGTH, TIME_TOL

    # ts = anns.nn.features[anns.nn.features.columns[-1]].values[:]]
    num_vals = len(anns.nn.features)
    if (anns.pu.debug ==anns.pu.on or anns.nn._DEBUG == anns.pu.on) and DEBUG_LOCAL != 0 and anns.myid == 0:
        print()
        for i in range(num_vals-SERIESLENGTH, num_vals):
            print('Before:(t,v)[{0}] = ({1}, {2})'.format(i,anns.elevTS.times[i],anns.elevTS.values[i]))

    ######################################################
    #SET UP nn BC from ADCIRC.
    ######################################################
    # Find the value of maximum depth first.
    if (anns.adcircrunflag != anns.pu.off):
        my_max_delta_eta = -1.0e+200
        my_min_delta_eta =  1.0e+200
        count=0.0
        my_avg_delta_eta=0.0
        my_eta_sum=0.0

        #for inode in range(1,anns.adcircedgestringnnodes+1):
        #    # -1 needed below since Python is 0 indexed whereas Fortan node numbers are 1-indexed
        #    node = anns.pb.nbvv[anns.adcircedgestringid][inode]-1
        for inode in range(anns.adcirc_cpld_elev_feedback_nnodes):
            # -1 needed below since Python is 0 indexed whereas Fortan node numbers are 1-indexed
            node = anns.adcirc_cpld_elev_feedback_nodes[inode]-1
            eta     = anns.pg.eta2[node]
            old_eta = anns.adcirc_hprev #Previous saved eta. #anns.pg.eta1[node]
            my_eta_sum       += eta
            # Gajanan gkc warning : These are only okay to use if the coupling time step is = adcirc time step!
            my_avg_delta_eta += eta - old_eta
            my_max_delta_eta = max(my_max_delta_eta, eta-old_eta)
            my_min_delta_eta = min(my_min_delta_eta, eta-old_eta)
            count += 1.0

        # Note: Hoping whichever depth is closes to NN current depth works better in damping oscillations than dmax alone!
        if (anns.pu.messg==anns.pu.on):
            eta_sum       = anns.pmsg.pymessg_dbl_sum(my_eta_sum      , anns.adcirc_comm_comp)
            avg_delta_eta = anns.pmsg.pymessg_dbl_sum(my_avg_delta_eta, anns.adcirc_comm_comp)
            max_delta_eta = anns.pmsg.pymessg_dbl_max(my_max_delta_eta, anns.adcirc_comm_comp)
            min_delta_eta = anns.pmsg.pymessg_dbl_min(my_min_delta_eta, anns.adcirc_comm_comp)
            count         = anns.pmsg.pymessg_dbl_sum(count           , anns.adcirc_comm_comp)
        else:
            eta_sum       = my_eta_sum
            avg_delta_eta = my_avg_delta_eta
            max_delta_eta = my_max_delta_eta
            min_delta_eta = my_min_delta_eta

        avg_eta = eta_sum/count
        #avg_delta_eta = avg_delta_eta/count # Previous time step
        avg_delta_eta    = avg_eta - anns.adcirc_hprev #Previous stopped ADCIRC time.
        anns.adcirc_hprev = avg_eta
        anns.adcirc_hprev_len = count

        if anns.pu.debug == anns.pu.on or DEBUG_LOCAL != 0:
            print("PE[{0}] Edge string({1}): Average eta =       {2}".format(anns.myid, anns.adcircedgestringid, avg_eta))
            print("PE[{0}] Edge string({1}): Maximum delta_eta = {2}".format(anns.myid, anns.adcircedgestringid, max_delta_eta))
            print("PE[{0}] Edge string({1}): Minimum delta_eta = {2}".format(anns.myid, anns.adcircedgestringid, min_delta_eta))
            print("PE[{0}] Edge string({1}): Average delta_eta = {2}".format(anns.myid, anns.adcircedgestringid, avg_delta_eta))

        # NOTE:
        # Unit of elevations in elevTS.values here is meters, same as that of
        # ADCIRC. Unit of time in elevTS.times is in seconds.
        # However, the neural network uses ft as units, so we must convert
        # meters to feet when setting the Neural Network's water elevations.
        # Also, the neural network time stamps and time steps are in hours, so
        # we must convert hours to seconds when setting time values.

        # Shift the time series and reset last accessed index
        anns.elevTS.times[:-1] = anns.elevTS.times[1:]
        anns.elevTS.values[:-1] = anns.elevTS.values[1:]
        #anns.elevTS.current_index = anns.elevTS.current_index-1

        ######################################################################################
        # Gajanan gkc. We need to decide what to use here. Stability is likely going to get
        # affected with this. Also important to consider that NN and ADCIRC may have
        # different values of depths at their interface.

        ## TYPE 1  -  This uses max/min depth.
        ## if abs(anns.elevTS.values[num_vals-2]-max_depth) < abs(anns.elevTS.values[num_vals-2]-min_depth):
        #if abs(max_delta_eta) < abs(min_delta_eta):
        #    delta_eta = max_delta_eta
        #else:
        #    delta_eta = min_delta_eta
        #anns.elevTS.values[num_vals-3] += delta_eta

        # TYPE 2  -  This uses average change in depth.
        anns.elevTS.values[num_vals-3] += avg_delta_eta

        ######################################################################################

        # Add the new value of time.
        if anns.couplingtype == 'ndAdn':
            DT = 0.0
            while (anns.nn.timer*anns.nn.timefact + DT < anns.adcirctprev+anns.adcircdt-TIME_TOL):
                DT += anns.effectivenndt
            anns.elevTS.times[num_vals-2] = anns.nn.timer*anns.nn.timefact + DT
        else:
            anns.elevTS.times[num_vals-2] = anns.adcirctprev

        # For round of errors:
        anns.elevTS.times[num_vals-1] = anns.elevTS.times[num_vals-2] + (TIME_TOL/anns.nn.timefact)
        anns.elevTS.values[num_vals-2] = anns.elevTS.values[num_vals-3]
        anns.elevTS.values[num_vals-1] = anns.elevTS.values[num_vals-2]

        # Set the values in the NN model.
        #print(anns.elevTS.current_index, anns.nn.timer*anns.nn.timefact)
        anns.elevTS._getTimeInterval(anns.nn.timer*anns.nn.timefact)
        #print(anns.nn.features[anns.nn.featurecols[-1]])
        value = anns.elevTS.interpolate(anns.nn.timer*anns.nn.timefact) # in meters
        #print(anns.elevTS.current_index, anns.nn.timer, value)
        anns.nn.features[anns.nn.featurecols[-1]].values[anns.nn.timer] = \
                value / anns.nn.length_factor # in feet.
        #print(anns.nn.features[anns.nn.featurecols[-1]])

    else:

        anns.elevTS.times[:-1] = anns.elevTS.times[1:]
        anns.elevTS.values[:-1] = anns.elevTS.values[1:]
        anns.elevTS.times[num_vals-1] += anns.elevTS.times[num_vals-3]-anns.elevTS.times[num_vals-4]

    if (anns.pu.debug == anns.pu.on or anns.nn._DEBUG == anns.pu.on) and DEBUG_LOCAL != 0 and anns.myid == 0:
        print('Current NN time = {0}'.format(anns.nn.timer*anns.nn.timefact))
        #print(anns.nn.features[anns.nn.featurecols[-1]])
        #assert(anns.nn.timer*anns.nn.timefact <= anns.elevTS.times[num_vals-2])
        #assert(anns.nn.timer*anns.nn.timefact >= anns.elevTS.times[0])
        for i in range(num_vals-SERIESLENGTH, num_vals):
            print('After :(t,v)[{0}] = ({1}, {2})'.format(i,anns.elevTS.times[i],anns.elevTS.values[i]))

############################################################################################################
if __name__ == '__main__':
    pass

