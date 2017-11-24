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

# Read
def read_mass_data(massdatadir, year):

    """
    Read in the mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :return: mass
    """

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

    return mass

def trim_mass_wxt_times(mass, WXT):

    """
    Trim the mass and WXT data based on their start and end times
    DOES NOT CHECK INTERNAL TIME MATCHING (BULK TRIM APPROACH)
    :return: mass
    :return: wxt
    """

    # Find start and end idx for mass and WXT
    if WXT['time'][0] < mass['time'][0]:
        wxt_start = np.where(WXT['time'] == mass['time'][0])[0][0]
        mass_start = 0
    else:
        wxt_start = 0
        mass_start = np.where(mass['time'] == WXT['time'][0])[0][0]
    # END
    if WXT['time'][-1] > mass['time'][-1]:
        wxt_end = np.where(WXT['time'] == mass['time'][-1])[0][0]
        mass_end = len(mass['time']) - 1
    else:
        wxt_end = len(WXT['time']) - 1
        mass_end = np.where(mass['time'] == WXT['time'][-1])[0][0]

    # create idx ranges where data exists for both mass and WXT
    wxt_range = np.arange(wxt_start, wxt_end, 1)
    mass_range = np.arange(mass_start, mass_end, 1)

    # trim data by only selecting the right time ranges
    for key, data in WXT.iteritems():
        WXT[key] = data[wxt_range]

    for key, data in mass.iteritems():
        mass[key] = data[mass_range]

    # returned data should have the same start and end times

    return mass, WXT

# Process

def calc_amm_sulph_and_amm_nit_from_gases(moles, mass):

    """
    Calculate the ammount of ammonium nitrate and sulphate from NH4, SO4 and NO3.
    Follows the CLASSIC aerosol scheme approach where all the NH4 goes to SO4 first, then to NO3.

    :param moles:
    :param mass:
    :return: mass [with the extra entries for the particles]
    """

    # define aerosols to make
    mass['(NH4)2SO4'] = np.empty(len(moles['SO4']))
    mass['(NH4)2SO4'][:] = np.nan
    mass['NH4NO3'] = np.empty(len(moles['SO4']))
    mass['NH4NO3'][:] = np.nan

    # calculate moles of the aerosols
    # help on GCSE bitesize:
    #       http://www.bbc.co.uk/schools/gcsebitesize/science/add_gateway_pre_2011/chemical/reactingmassesrev4.shtml
    for i in range(len(moles['SO4'])):
        if moles['SO4'][i] > (moles['NH4'][i] / 2):  # more SO4 than NH4 (2 moles NH4 to 1 mole SO4) # needs to be divide here not times

            # all of the NH4 gets used up making amm sulph.
            mass['(NH4)2SO4'][i] = mass['NH4'][i] * 7.3  # ratio of molecular weights between amm sulp and nh4
            # rem_nh4 = 0

            # no NH4 left to make amm nitrate
            mass['NH4NO3'][i] = 0
            # some s04 gets wasted
            # rem_SO4 = +ve

        # else... more NH4 to SO4
        elif moles['SO4'][i] < (moles['NH4'][i] / 2):  # more NH4 than SO4 for reactions

            # all of the SO4 gets used in reaction
            mass['(NH4)2SO4'][i] = mass['SO4'][i] * 1.375  # ratio of SO4 to (NH4)2SO4
            # rem_so4 = 0

            # some NH4 remains this time!
            # remove 2 * no of SO4 moles used from NH4 -> SO4: 2, NH4: 5; therefore rem_nh4 = 5 - (2*2)
            rem_nh4 = moles['NH4'][i] - (moles['SO4'][i] * 2)

            if moles['NO3'][i] > rem_nh4:  # if more NO3 to NH4 (1 mol NO3 to 1 mol NH4)

                # all the NH4 gets used up
                mass['NH4NO3'][i] = rem_nh4 * 4.4  # ratio of amm nitrate to remaining nh4
                # rem_nh4 = 0

                # left over NO3
                # rem_no3 = +ve

            elif moles['NO3'][i] < rem_nh4:  # more remaining NH4 than NO3

                # all the NO3 gets used up
                mass['NH4NO3'][i] = mass['NO3'][i] * 1.29
                # rem_no3 = 0

                # some left over nh4 still
                # rem_nh4_2ndtime = +ve

    return mass

def internal_time_completion(data, date_range):

    """
    Set up new dictionary for data with a complete time series (no gaps in the middle).
    :param: data (must be a dictionary with a list of datetimes with the keyname 'time', i.e. data['time'])
    :return: data_full

    Done by checking if time_i from the complete date range (with no time gaps) exists in the data, and if so, extract
    out the values and put it into the new dictionary.
    """


    # set up temporary dictionaries for data (e.g. data_full) with empty arrays for each key, ready to be filled
    data_full = {}
    for h in data.iterkeys():
        data_full[h] = np.empty(len(date_range))
        data_full[h][:] = np.nan

    # replace time with date range
    data_full['time'] = date_range

    # step through time and time match data for extraction
    for t, time_t in enumerate(date_range):
        idx = np.where(data['time'] == time_t)[0]

        # if not empty, put in the data to new array
        if idx.size != 0:
            for h in data.iterkeys():
                data_full[h][t] = data[h][idx]

    return data_full

def convert_mass_to_kg_kg(mass, WXT, aer_particles):

    """
    Convert mass molecules from g m-3 to kg kg-1

    :param mass
    :param WXT (for meteorological data)
    :param aer_particles (not all the keys in mass are the species, therefore only convert the species defined above)
    :return: mass_kg_kg: mass in kg kg-1 air
    """

    # convert temperature to Kelvin
    T_K = WXT['Tair'] + 273.15
    p_Pa = WXT['press'] * 100.0

    # density of air [kg -3] # assumes dry air atm
    # p = rho * R * T [K]
    WXT['dryair_rho'] = p_Pa / (286.9 * T_K)

    # convert g m-3 air to kg kg-1 of air
    mass_kg_kg = {'time': mass['time']}
    for aer_i in aer_particles:
        mass_kg_kg[aer_i] = mass[aer_i] * 1e-3 / WXT['dryair_rho']

    return mass_kg_kg, WXT

# Swelling

def calc_r_md_species(r_d_microns, WXT, aer_i):

    """
    Calculate the r_md for all particles, given the RH and what species

    :param r_d_microns:
    :param WXT: (needed for RH)
    :param aer_i:
    :return: r_md

    Currently just works for ammonium sulphate, ammonium nitrate and NaCl
    """

    # calulate r_md based on Fitzgerald (1975) eqn 8 - 10
    def calc_r_md_i(rh_i, alpha_factor):

        """
        Calculate r_md for a single value of rh (rh_i)
        :param rh_i:
        :return: r_md_i
        """

        beta = np.exp((0.00077 * rh_i) / (1.009 - rh_i))
        if rh_i < 0.97:
            phi = 1.058 - ((0.0155 * (rh_i - 0.97))
                           / (1.02 - (rh_i ** 1.4)))
        else:
            phi = 1.058
        alpha = 1.2 * np.exp((0.066 * rh_i) / (phi - rh_i))

        # alpha factor comes from the Table 1 in Fitzgerald (1975) to be used with some other aerosol types
        r_md_i = alpha_factor * alpha * (r_d_microns ** beta)

        return r_md_i


    # Set up array for aerosol
    r_md =  np.empty(len(WXT['time']))
    r_md[:] = np.nan

    phi = np.empty(len(WXT['time']))
    phi[:] = np.nan

    # limits for what approach to use, depending on the RH
    # from the CLASSIC guidence, follows Fitzgerald (1975)
    if aer_i == '(NH4)2SO4':
        rh_cap = 0.995 # calculate r_md specifically for the upper limit (considered max rh)
        rh_del = 0.81 # calculate r_md specifically for the upper limit (start of empirical formula)
                     # CLASSIC does linear interpolation bettween rh_del and rh_eff.
        rh_eff = 0.3 # efflorescence (below is dry)
        alpha_factor = 1.0 # a coefficient for alpha, which is specific for different aerosol types
    elif aer_i == 'NH4NO3':
        rh_cap = 0.995
        rh_del = 0.61
        rh_eff = 0.3
        alpha_factor = 1.06

    elif aer_i == 'NaCl':
        rh_cap = 0.995
        rh_del = 0.75
        rh_eff = 0.42
        alpha_factor = 1.35

    # --------------------------------------------
    # Calculate r_md for the species, given RH
    # -----------------------------------------------

    # empirical relationships fitted for radius in micrometers, not meters (according to CLASSIC guidance).

    # --- delequescence - rh cap (defined as 0.995. Above this empirical relationship breaks down) --- #

    # Currently just calculates it for all, then gets overwritten lower down, depending on their RH (e.g. below eff)
    # ToDo use the rh_bet_del_cap to only calc for those within the del - cap range.

    # # between deliquescence and rh_cap (set at 0.995 for all)
    # bool = np.logical_and(WXT['RH_frac'] >= rh_del, WXT['RH_frac'] <= rh_cap)
    # rh_bet_del_cap = np.where(bool == True)[0]

    beta = np.exp((0.00077 * WXT['RH'])/(1.009 - WXT['RH']))
    rh_lt_97 = WXT['RH_frac'] < 0.97
    phi[rh_lt_97] = 1.058
    phi[~rh_lt_97] = 1.058 - ((0.0155 * (WXT['RH_frac'][~rh_lt_97] - 0.97))
                              /(1.02 - (WXT['RH_frac'][~rh_lt_97] ** 1.4)))
    alpha = 1.2 * np.exp((0.066 * WXT['RH_frac'])/ (phi - WXT['RH_frac']))

    r_md = alpha_factor * alpha * (r_d_microns ** beta)

    # --- above rh_cap ------#

    # set all r_md(RH>99.5%) to r_md(RH=99.5%) to prevent growth rates inconsistent with impirical equation.
    # replace all r_md values above 0.995 with 0.995
    rh_gt_cap = WXT['RH_frac'] > rh_cap
    r_md[rh_gt_cap] = calc_r_md_i(rh_cap, alpha_factor)

    # --- 0 to efflorescence --- #

    # below efflorescence point (0.3 for sulhate, r_md = r_d)
    rh_lt_eff = WXT['RH_frac'] <= rh_eff
    r_md[rh_lt_eff] = r_d_microns

    # ------ efflorescence to deliquescence ----------#

    # calculate r_md for the deliquescence rh - used in linear interpolation
    r_md_del = calc_r_md_i(rh_del, alpha_factor)

    # all values that need to have some linear interpolation
    bool = np.logical_and(WXT['RH_frac'] >= rh_eff, WXT['RH_frac'] <= rh_del)
    rh_bet_eff_del = np.where(bool == True)[0]

    # between efflorescence point and deliquescence point, r_md is expected to value linearly between the two
    low_rh = rh_eff
    up_rh = rh_del
    low_r_md = r_d_microns
    up_r_md = r_md_del

    diff_rh = up_rh - low_rh
    diff_r_md = r_md_del - r_d_microns
    abs_diff_r_md = abs(diff_r_md)

    # find distance rh is along linear interpolation [fraction] from lower limit
    # frac = np.empty(len(r_md))
    # frac[:] = np.nan
    frac = ((WXT['RH_frac'][rh_bet_eff_del] - low_rh) / diff_rh)

    # calculate interpolated values for r_md
    r_md[rh_bet_eff_del] = low_r_md + (frac * abs_diff_r_md)

    return r_md

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

    # aerosol particles to calculate (OC = Organic carbon, CBLK = black carbon, both already measured)
    # match dictionary keys further down
    aer_particles = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK']

    # density of molecules [kg m-3]
    # CBLK: # Zhang et al., (2016) Measuring the morphology and density of internally mixed black carbon
    #           with SP2 and VTDMA: New insight into the absorption enhancement of black carbon in the atmosphere
    # ORG: Range of densities for organic carbon is mass (0.625 - 2 g cm-3)
    #  Haywood et al 2003 used 1.35 g cm-3 but Schkolink et al., 2006 claim the average is 1.1 g cm-3 after a lit review
    aer_density = {'(NH4)2SO4': 1770.0,
                   'NH4NO3': 1720.0,
                   'NaCl': 2160.0,
                   'CORG': 1100.0,
                   'CBLK': 1200.0}

    # pure water density
    water_density = 1000.0 # kg m-3

    # ==============================================================================
    # Read data
    # ==============================================================================

    # Read in species by mass data
    # Units are grams m-3
    mass_in = read_mass_data(massdatadir, year)


    # Read WXT data
    wxtfilepath = wxtdatadir + wxt_inst_site + '_' + year + '_15min.nc'
    WXT_in = eu.netCDF_read(wxtfilepath, vars=['RH', 'Tair','press', 'time'])
    WXT_in['RH_frac'] = WXT_in['RH'] * 0.01
    WXT_in['time'] -= dt.timedelta(minutes=15) # change time from 'obs end' to 'start of obs', same as the other datasets

    # Trim times
    # as WXT and mass data are 15 mins and both line up exactly already
    #   therefore trim WXT to match mass time
    mass_in, WXT_in = trim_mass_wxt_times(mass_in, WXT_in)

    # Time match so mass and WXT times line up INTERNALLY as well
    date_range = eu.date_range(WXT_in['time'][0], WXT_in['time'][-1], 15, 'minutes')

    # make sure there are no time stamp gaps in the data so mass and WXT will match up perfectly, timewise.
    print 'beginning time matching for WXT...'
    WXT = internal_time_completion(WXT_in, date_range)
    print 'end time matching for WXT...'

    # same but for mass data
    print 'beginning time matching for mass...'
    mass = internal_time_completion(mass_in, date_range)
    print 'end time matching for mass...'

    # ==============================================================================
    # Process data
    # ==============================================================================

    # molecular mass of each molecule
    mol_mass_amm_sulp = 132
    mol_mass_amm_nit = 80
    mol_mass_nh4 = 18
    mol_mass_n03 = 62
    mol_mass_s04 = 96

    # Convert into moles
    # calculate number of moles (mass [g] / molar mass)
    # 1e-06 converts from micrograms to grams.
    moles = {'SO4': mass['SO4'] / mol_mass_s04,
             'NO3': mass['NO3'] / mol_mass_n03,
             'NH4': mass['NH4'] / mol_mass_nh4}


    # calculate ammonium sulphate and ammonium nitrate from gases
    # adds entries to the existing dictionary
    mass = calc_amm_sulph_and_amm_nit_from_gases(moles, mass)

    # convert chlorine into sea salt assuming all chlorine is sea salt, and enough sodium is present.
    #      potentially weak assumption for the chlorine bit due to chlorine depletion!
    mass['NaCl'] = mass['CL'] * 1.65

    # convert masses from g m-3 to kg kg-1_air for swelling.
    # Also creates the air density and is stored in WXT
    mass_kg_kg, WXT = convert_mass_to_kg_kg(mass, WXT, aer_particles)

    # start with just 0.11 microns as the radius - can make it more fancy later...
    r_d_microns = 0.11  # [microns]
    r_d_m = r_d_microns * 1.0e-6  # [m]
    V_dry_from_r_d = (4.0/3.0) * np.pi * (r_d_m ** 3.0)  # [m3]

    # calculate the number of particles for each species using radius_m and the mass
    # Hopefull not needed!
    num_part = {}
    for aer_i in aer_particles:
        num_part[aer_i] = mass_kg_kg[aer_i] / ((4.0/3.0) * np.pi * (aer_density[aer_i]/WXT['dryair_rho']) * (r_d_m ** 3.0))

    # calculate dry volume
    V_dry_from_mass = {}
    for aer_i in aer_particles:
        # V_dry[aer_i] = (4.0/3.0) * np.pi * (r_d_m ** 3.0)
        V_dry_from_mass[aer_i] = mass_kg_kg[aer_i] / aer_density[aer_i]  # [m3]

    # ---------------------------------------------------------
    # Swell the particles (r_md,aer_i) [microns]

    # set up dictionary
    r_md = {}

    # calculate the swollen particle size for these three aerosol types
    # Follows CLASSIC guidence, based off of Fitzgerald (1975)
    for aer_i in ['(NH4)2SO4', 'NH4NO3', 'NaCl']:
        r_md[aer_i] = calc_r_md_species(r_d_microns, WXT, aer_i)

    # set r_md for black carbon as r_d, assuming black carbon is completely hydrophobic
    r_md['CBLK'] = np.empty(len(date_range))
    r_md['CBLK'][:] = r_d_microns

    # calculate r_md for organic carbon using the MO empirically fitted g(RH) curves
    # r_md['CORG'] = ...


    # -----------------------------------------------------------

    # calculate abs volume of wetted particles (V_abs,wet,aer_i)
    # use the growth factors calculated based on r_d to calc V_wet from V_dry(mass, density)

    # calculate the physical growth factor, wetted particle density, wetted particle volume, ...
    #   and water volume (V_wet - Vdry)

    GF = {}
    aer_wet_density = {}
    V_wet_from_mass = {}
    V_water_i  = {}
    for aer_i in ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CBLK']: # aer_particles:

        # physical growth factor
        GF[aer_i] = r_md[aer_i] / r_d_microns

        # wet aerosol density
        aer_wet_density[aer_i] = (aer_density[aer_i] / (GF[aer_i]**3.0)) + \
                                 (water_density * (1.0 - (1.0 / (GF[aer_i]**3.0))))


        # wet volume, using the growth rate from r_d to r_md
        V_wet_from_mass[aer_i] = V_dry_from_mass[aer_i] * (GF[aer_i] ** 3.0)


        # if np.nan (i.e. there was no mass therefore no volume) make it 0.0
        bin = np.isnan(V_wet_from_mass[aer_i])
        V_wet_from_mass[aer_i][bin] = 0.0

        # water volume contribution from just this aer_i
        V_water_i[aer_i] = V_wet_from_mass[aer_i] - V_dry_from_mass[aer_i]

    # calculate total water volume
    V_water_2d = np.array(V_water_i.values()) # turn into a 2D array
    V_water_tot = np.nansum(V_water_2d, axis=0)

    # combine volumes of the DRY aerosol and the water into a single 2d array shape=(time, substance)
    V_abs = np.transpose(np.vstack([np.array(V_dry_from_mass.values()),V_water_tot]))

    # now calculate the relative volume of each of these
    # scale absolute volume to find relative volume of each (such that sum(all substances for time t = 1))
    scaler = 1.0/(np.nansum(V_abs, axis=1)) # a value for each time step
    scaler_rep = np.transpose(np.array([scaler]*6)) # 2d array shape=(time, substance)
    V_rel = scaler_rep * V_abs

    # --------------------------------------------------------------

    # calculate n_mixed using volume mixing method




    # calculate relative volume of all wet aerosol (V_rel,wet,aer_i)


    # calculate n_mixed using volume mixing and (V_rel,wet,aer_i)


    # calculate volume mean radii from r_md,aer_i (weighted by V_rel,wet,aer_i)


    # calculate Q_back and Q_ext from the avergae r_md and n_mixed


    # calculate lidar ratio


    # assume radii are 0.11 microns for now...






    return

if __name__ == '__main__':
    main()

print 'END PROGRAM'