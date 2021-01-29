#!/usr/bin/env python
#------------------------------------------------------------------------------#
# adcirc-nn - Software for physics-based machine learning with ADCIRC
# LICENSE: BSD 3-Clause "New" or "Revised"
#------------------------------------------------------------------------------#

################################################################################
DEBUG_LOCAL = 1

################################################################################
def nn_init_bc_from_adcirc_depths(anns): # anns is of type adcircnnstruct.
    from .adcirc_nn_class import SERIESLENGTH
    assert(len(anns.nn.features) >= SERIESLENGTH)

    ######################################################
    #SET UP NN BC from ADCIRC.
    ######################################################
    # Find the value of maximum depth first.
    eta=0.0
    my_eta_sum =  0.0
    my_max_eta = -1.0e+200
    my_min_eta =  1.0e+200
    count=0.0

    print('Coupled nnodes = ', anns.adcirc_cpld_elev_feedback_nnodes)
    print('Coupled nodes = [', anns.adcirc_cpld_elev_feedback_nodes, ']')
    for inode in range(anns.adcirc_cpld_elev_feedback_nnodes):
        # -1 needed below since Python is 0 indexed whereas Fortan node numbers are 1-indexed
        node = anns.adcirc_cpld_elev_feedback_nodes[inode]-1
        eta = anns.pg.eta2[node]
        my_eta_sum += eta
        my_max_eta = max(my_max_eta, eta)
        my_min_eta = min(my_min_eta, eta)
        count += 1.0

    # Note: Hoping whichever depth is closes to NN current depth works better in damping oscillations than dmax alone!
    if (anns.pu.messg==anns.pu.on):
        eta_sum = anns.pmsg.pymessg_dbl_sum(my_eta_sum, anns.adcirc_comm_comp)
        max_eta = anns.pmsg.pymessg_dbl_max(my_max_eta, anns.adcirc_comm_comp)
        min_eta = anns.pmsg.pymessg_dbl_min(my_min_eta, anns.adcirc_comm_comp)
        count   = anns.pmsg.pymessg_dbl_sum(count     , anns.adcirc_comm_comp)
    else:
        eta_sum = my_eta_sum
        max_eta = my_max_eta
        min_eta = my_min_eta
    avg_eta = eta_sum/count
    anns.adcirc_hprev = avg_eta # Going to be taking the average.
    anns.adcirc_hprev_len = count # Going to be taking the average.
    print("PE[{0}] Edge string({1}): Starting Average eta2 = {2}, count = {3}".format(anns.myid,anns.adcircedgestringid,anns.adcirc_hprev,anns.adcirc_hprev_len))


    ######################################################
    num_vals = len(anns.nn.features)
    if (anns.pu.debug == anns.pu.on or anns.nn._DEBUG == anns.pu.ON) and DEBUG_LOCAL != 0 and anns.myid == 0:
        print('\nSetting up Boundary time series for NN.')
        for i in range(num_vals-SERIESLENGTH, num_vals):
            print('Before:(t,v)[{0}] = ({1}, {2})'.format(i,anns.elevTS.times[i],anns.elevTS.values[i]))

    superdt = 0.0
    while (superdt < anns.adcirctstart + anns.adcircdt):
        superdt += anns.effectivenndt
        #superdt += max(60.0, anns.effectivenndt)
    superdt = int(superdt/anns.nn.timefact+1.0e-6)

    # Set the time stamps in the time series
    for i in range(num_vals):
        anns.elevTS.times[i] = (anns.nn.timer - (num_vals-2-i)*superdt)*anns.nn.timefact
    if anns.couplingtype == 'ndAdn':
        anns.elevTS.times += superdt*anns.nn.timefact
    anns.elevTS.times[num_vals-1] = anns.elevTS.times[num_vals-2] + anns.effectivenndt

    # Set the values in the time series to the first value.
    # This is not done in the NN features variable (see below).
    anns.elevTS.values[:] = anns.elevTS.values[0]

    # NOTE:
    # Unit of elevations in elevTS.values here is meters, same as that of
    # ADCIRC. Unit of time in elevTS.times is in seconds.
    # However, the neural network uses ft as units, so we must convert
    # meters to feet when setting the Neural Network's water elevations.
    # Also, the neural network time stamps and time steps are in hours, so
    # we must convert hours to seconds when setting time values.
    # The above note is a copy of the note in nn_set_bc_func.py.

    # Set the values in the NN model.
    #print(anns.elevTS.current_index, anns.nn.timer*anns.nn.timefact)
    anns.elevTS._getTimeInterval(anns.nn.timer*anns.nn.timefact)
    value = anns.elevTS.interpolate(anns.nn.timer*anns.nn.timefact) # in meters
    #print(anns.elevTS.current_index, anns.nn.timer, value)
    #print(anns.nn.features[anns.nn.featurecols[-1]])
    anns.nn.features[anns.nn.featurecols[-1]].values[anns.nn.timer] = \
            value / anns.nn.length_factor # in ft.
    #print(anns.nn.features[anns.nn.featurecols[-1]].values)
    #print(anns.nn.features[anns.nn.featurecols[-1]])

    if (anns.pu.debug == anns.pu.on or anns.nn._DEBUG == anns.pu.ON) and DEBUG_LOCAL != 0 and anns.myid == 0:
        print('Current NN time = {0}'.format(anns.nn.timer*anns.nn.timefact))
        #print(anns.nn.features[anns.nn.featurecols[-1]])
        for i in range(num_vals-SERIESLENGTH, num_vals):
            print('After :(t,v)[{0}] = ({1}, {2})'.format(i,anns.elevTS.times[i],anns.elevTS.values[i]))

    #exit()

################################################################################
if __name__ == '__main__':
    pass

