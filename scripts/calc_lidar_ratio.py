"""
Read in mass data from London and calculate the Lidar ratio using the CLASSIC scheme approach, from Ben's help.

Created by Elliott Fri 17 Nov 2017
"""

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pickle

import numpy as np
import datetime as dt

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

def calc_amm_sulph_and_amm_nit_from_gases(moles, mass):

    """
    Calculate the ammount of ammonium nitrate and sulphate from NH4, SO4 and NO3.
    Follows the CLASSIC aerosol scheme approach where all the NH4 goes to SO4 first, then to NO3.

    :param moles:
    :param mass:
    :return:
    """

    # calculate moles of the aerosols
    # help on GCSE bitesize:
    #       http://www.bbc.co.uk/schools/gcsebitesize/science/add_gateway_pre_2011/chemical/reactingmassesrev4.shtml
    if moles['SO4'] > (moles['NH4'] / 2):  # more SO4 than NH4 (2 moles NH4 to 1 mole SO4) # needs to be divide here not times

        # all of the NH4 gets used up making amm sulph.
        mass['(NH4)2SO4'] = mass['NH4'] * 7.3  # ratio of molecular weights between amm sulp and nh4
        # rem_nh4 = 0

        # no NH4 left to make amm nitrate
        mass['NH4N03'] = 0
        # some s04 gets wasted
        # rem_SO4 = +ve

    # else... more NH4 to SO4
    elif moles['SO4'] < (moles['NH4'] / 2):  # more NH4 than SO4 for reactions

        # all of the SO4 gets used in reaction
        mass['(NH4)2SO4'] = mass['SO4'] * 1.375  # ratio of SO4 to (NH4)2SO4
        # rem_so4 = 0

        # some NH4 remains this time!
        # remove 2 * no of SO4 moles used from NH4 -> SO4: 2, NH4: 5; therefore rem_nh4 = 5 - (2*2)
        rem_nh4 = moles['NH4'] - (moles['SO4'] * 2)

        if moles['NO3'] > rem_nh4:  # if more NO3 to NH4 (1 mol NO3 to 1 mol NH4)

            # all the NH4 gets used up
            mass['NH4NO3'] = rem_nh4 * 4.4  # ratio of amm nitrate to remaining nh4
            # rem_nh4 = 0

            # left over NO3
            # rem_no3 = +ve

        elif moles['NO3'] < rem_nh4:  # more remaining NH4 than NO3

            # all the NO3 gets used up
            mass['NH4NO3'] = mass['NO3'] * 1.29
            # rem_no3 = 0

            # some left over nh4 still
            # rem_nh4_2ndtime = +ve

    return moles


def main():


    # Read in the mass data for 2016
    # Read in RH data for 2016
    # convert gases and such into the aerosol particles
    # swell the particles based on the CLASSIC scheme stuff
    # use Mie code to calculate the backscatter and extinction
    # calculate lidar ratio
    # plot lidar ratio

    # ==============================================================================
    # Setup
    # ==============================================================================

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/'
    massdatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/ERG/'


    savedir = maindir + 'figures/LidarRatio/'

    # data
    wxtdatadir = datadir + 'L1/'

    # RH data
    wxt_inst_site = 'WXT_KSSW'

    # data year
    year = '2016'

    # ==============================================================================
    # Read data
    # ==============================================================================

    # Read in species by mass data
    # Units are micro grams m-3
    massfname = 'PM_North_Kensington_'+year+'.csv'
    massfilepath = massdatadir + massfname
    massrawData = np.genfromtxt(massfilepath, delimiter=',', dtype="|S20") # includes the header

    mass = {'time': np.array([dt.datetime.strptime(i[0], '%d/%m/%Y %H:%M') for i in massrawData[1:]])}

    # get headers without the site part of it (S04@NK to S04)
    headers = [i.split('@')[0] for i in massrawData[0][1:]]

    # ignore first entry, as that is the date&time
    for h, header_site in enumerate(massrawData[0][1:]):

        # get the main part of the header from the
        split = header_site.split('@')
        header = split[0]

        # turn '' into nans
        # convert from micrograms to grams
        mass[header] = np.array([np.nan if i[h+1] == '' else i[h+1] for i in massrawData[1:]], dtype=float) * 1e-06


    # QAQC - turn all negative values in each column into nans if one of them is negative
    for header_i in headers:
        idx = np.where(mass[header_i] < 0.0)
        for header_j in headers:
            mass[header_j][idx] = np.nan


    # Read RH data
    wxtfilepath = wxtdatadir + wxt_inst_site + '_' + year + '_15min.nc'
    WXT = eu.netCDF_read(wxtfilepath, vars=['RH', 'Tair','press', 'time'])
    WXT['time'] -= dt.timedelta(minutes=15) # change time from 'obs end' to 'start of obs', same as the other datasets


    # Process the variables

    # Convert into moles

    mol_mass_amm_sulp = 132
    mol_mass_amm_nit = 80
    mol_mass_nh4 = 18
    mol_mass_n03 = 62
    mol_mass_s04 = 96

    # calculate number of moles (mass [g] / molar mass)
    # 1e-06 converts from micrograms to grams.
    moles = {'SO4': mass['SO4'] / mol_mass_s04,
             'NO3': mass['NO3'] / mol_mass_n03,
             'NH4': mass['NH4'] / mol_mass_nh4}


    # calculate ammonium sulphate and ammonium nitrate from gases
    moles = calc_amm_sulph_and_amm_nit_from_gases(moles)

    # convert chlorine into sea salt assuming all chlorine is sea salt, and enough sodium is present.
    #      potentially weak assumption for the chlorine bit due to chlorine depletion!
    mass['NaCl'] = mass['CL'] * 1.65

    # convert masses from g m-3 to kg kg-1_air for swelling.
    # use observed Tair and pressure from KSSW WXT to calculate air density





    # assume radii are 0.11 microns for now...






    return

if __name__ == '__main__':
    main()

print 'END PROGRAM'