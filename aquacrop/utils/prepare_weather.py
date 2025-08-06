#-----------------------------------------------------------------------------------------
# CHANGES IMPLEMENTED BY KADE FLYNN AUGUST 2025
#-----------------------------------------------------------------------------------------
#- Add function prepare_weather_from_daymet() to format weather data downloaded from Daymet (https://daymet.ornl.gov/) for use in Aquacrop. As part of this function, reference ET is calculated...
#- Add function to calculate Reference ET according to Allen (1998). Wind speed data is   assumed to be equal to 2 m/s.
#- Add functions to calculate net radiaition from input weather data. Net radiation calculation is based on equations in An Introduction to Environmental Biophysics (Campbell and Norman, 1998). Equations include:
#   - calculate_net_radiation()
#   - _calculate_saturated_vapor_pressure()
#   - _calculate_clear_sky_emissivity()
#   - _calculate_total_radiant_energy_emmitted()
#   - _calculate_max_radiation
#   - _calculate_cloud_correction()
#-----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np  # required for changes implemented by kade flynn AUgust 2025


def prepare_weather(weather_file_path):
    """
    function to read in weather data and return a dataframe containing
    the weather data

    Arguments:\n

FileLocations): `FileLocationsClass`:  input File Locations

weather_file_path): `str):  file location of weather data



    Returns:

weather_df (pandas.DataFrame):  weather data for simulation period

    """

    weather_df = pd.read_csv(weather_file_path, header=0, sep='\s+')

    assert len(weather_df.columns) == 7

    # rename the columns
    weather_df.columns = str(
        "Day Month Year MinTemp MaxTemp Precipitation ReferenceET").split()

    # put the weather dates into datetime format
    weather_df["Date"] = pd.to_datetime(weather_df[["Year", "Month", "Day"]])

    # drop the day month year columns
    weather_df = weather_df.drop(["Day", "Month", "Year"], axis=1)

    # set limit on ET0 to avoid divide by zero errors
    weather_df['ReferenceET'] = weather_df['ReferenceET'].clip(lower=0.1)

    return weather_df

def prepare_weather_from_daymet(weather_file_path: str, latitude: float, longitude: float, elevation: float):
    """
    function to read in weather data from DAYMET and return a dataframe containing the weather data.

    Arguments:
    - file path to daymmet download. File structure:
        - Column names begin on row 7
        - Columns are:
            - year, yday, dayl (s), srad (W/m^2), swe (kg/m^2), tmax (deg c) tmin (deg c), vp(Pa)
    - latitude
    -longitude
    -elevation

    Returns:
    - weather_df (pandas.DataFrame):  weather data for simulation period

    """

    weather_df = pd.read_csv(weather_file_path, skiprows=6, header=0)

    assert len(weather_df.columns) == 9

    # rename the columns
    weather_df.columns = str(
        "Year Yday Dayl_s Precip_mm Srad_wm2 Swe_kgm2 Tmax_c Tmin_c Vp_pa").split()

    # put the weather dates into datetime format
    weather_df['Date_Yj'] = pd.to_datetime(weather_df['Year'].astype(str) + '-' + weather_df['Yday'].astype(str), format='%Y-%j')
    weather_df['Date'] = weather_df['Date_Yj'].dt.strftime('%Y-%m-%d')

    # drop the  date in year-yday format columns
    weather_df = weather_df.drop(["Date_Yj"], axis=1)

    # calculate incoming shortwave (solar) radiation
    weather_df['Solar_mjm2'] = weather_df['Srad_wm2'] * weather_df['Dayl_s'] / 1_000_000

    # drop shortwave radiation flux density, day length, snow water equivalent
    weather_df = weather_df.drop(['Srad_wm2', 'Dayl_s', 'Swe_kgm2'], axis=1)

    # Calculate Net Radiation
    weather_df['Rn'] = calculate_net_radiation(Tmin = weather_df['Tmin_c'],
                                               Tmax = weather_df['Tmax_c'],
                                               Solar = weather_df['Solar_mjm2'],
                                               Vp = weather_df['Vp_pa'],
                                               Yday = weather_df['Yday'],
                                               Latitude = latitude,
                                               Longitude = longitude,
                                               Elevation = elevation)
    # Calculate Reference ET
    weather_df['ReferenceET'] = calculate_reference_et()

    # set limit on ET0 to avoid divide by zero errors
    #weather_df['ReferenceET'] = weather_df['ReferenceET'].clip(lower=0.1)

    return weather_df


def calculate_reference_et(Rn, Vp):
    '''Calculate Reference ET with the FAO Penman-Monteith equation (Equation 3, Allen et al. 1998).
    
    Arguments:
    -net radiation
    -soil heat flux
    -vapor pressure
    -saturated vapor pressure
    -minimum daily temperatuer
    -maximum daily temperature
    -psychrometric constant 
    '''
    numerator

def calculate_net_radiation(Tmin, Tmax, Solar, Vp, Yday, Latitude, Longitude, Elevation):
    '''
    Function to calculate the net radiation. General reference is An Introduction to Environmental Biophysics (Campbell and Norman, 1998).

    Parameters:
    -----------
    Tmin : float
        Minimum daily temperature in degrees celcius
    Tmax : float
        Maximum daily temperature in degrees celsius
    Solar: float
        Incoming shortwave (solar) radiation in Mj/m^2
    Vp : float
        Vapor pressure in Pascales 
    Latitude: float
        latitude in degrees
    Longitude: float
        longitude in degrees
    
    Result:
    -------
    NetRadiation
        Object containing the net radiation and related variables
    '''
    Temperature = (Tmin + Tmax) / 2

    saturated_vapor_pressure, slope_saturated_vapor_pressure_curve = _calculate_saturated_vapor_pressure(temperature = Temperature)

    clear_sky_emissivity = _calculate_clear_sky_emissivity(temperature = Temperature, vapor_pressure = Vp)

    total_radiant_energy_emitted = _calculate_total_radiant_energy_emitted(temperature = Temperature, clear_sky_emissivity = clear_sky_emissivity)
    
    max_radiation, day_length = _calculate_max_radiation(latitude = Latitude, longitude = Longitude, standard_meridian = Longitude, elevation = Elevation, day_of_year = Yday, solar_radiation = Solar)
    
    cloud_correction = _calculate_cloud_correction(solar_radiation = Solar, max_radiation = max_radiation)

    albedo = 0.23
    net_radiation = Solar * (1-albedo) + total_radiant_energy_emitted * cloud_correction

    return net_radiation

def _calculate_saturated_vapor_pressure(temperature):
    '''
    Function to calculate the saturated vapor pressure and the slope of the saturated vapor pressure curve
    using Tetens formula. Equation 3.8 from An Introduction to Environmental Biophysics (Campbell and Norman, 1998).
    '''
    a = 0.611
    b = 17.502
    c = 240.970
    saturated_vapor_pressure = a * (np.exp(b*temperature/(temperature+c)))

    slope_saturated_vapor_pressure_curve = b*c*saturated_vapor_pressure/\
        ((temperature+c)**2) / 101.3

    return saturated_vapor_pressure, slope_saturated_vapor_pressure_curve

def _calculate_clear_sky_emissivity(temperature: int, vapor_pressure: int):
    '''
    Function to calculate the clear sky emissivity. Equation 10.10 from An Introduction to
    Environmental Biophysics (Campbell and Norman, 1998).
    '''

    clear_sky_emissivity = 1.72 * ((vapor_pressure/1000)/(temperature+273))**(1/7)

    return clear_sky_emissivity

def _calculate_total_radiant_energy_emitted(temperature: int, clear_sky_emissivity: int):
    '''
    Function to calculate the total radiant energy emitted by a blackbody. Equation 10.9 from An Introduction to
    Environmental Biophysics (Campbell and Norman, 1998).
    '''

    total_radiant_energy_emitted = 0.98 * (clear_sky_emissivity-1) * 5.67e-8 * (temperature+273)**4
    # convert to Mj/m2. Output of Stefan-Boltzmann law is W/m2
    total_radiant_energy_emitted = total_radiant_energy_emitted * 14 * 3600 * 1e-6

    return total_radiant_energy_emitted


def _calculate_max_radiation(latitude: int, longitude: int, standard_meridian: int,
                              elevation: int, day_of_year: int, solar_radiation: int):
    '''
    Function to calculate the maximum radiation for a given location. General reference
    is An Introduction to Environmental Biophysics (Campbell and Norman, 1998). pages 168 to 173.
    '''
    # extraterrestrial flux density normal to the solar beam [W/m^2]
    spo = 1360
    # atmoshperic transmittance
    tao = 0.75
    # calculate atmospheric pressure
    pa = 101.3 * np.exp(-elevation/8200)
    # Equation of time (Eq. 11.4)
    f = 279.575 + 0.9856 * day_of_year
    # convert f from radians to degrees
    f = f * (np.pi / 180)

    et = (-104.7 * np.sin((f)) + 596.2 * np.sin((2*f)) + 4.3*np.sin((3*f)) -\
          12.7*np.sin((4*f)) - 429.3*np.cos((f)) - 2*np.cos((2*f)) +\
            19.3*np.cos((3*f))) / 3600
    # solar declination angle in degrees (Eq. 11.5)
    sigma = (180/np.pi) * (np.asin(0.39785*np.sin((np.pi/180)*(278.97 + 0.9856*day_of_year + 1.9165*\
                                                                   np.sin((np.pi/180)*(356.6+0.9856*day_of_year))))))
    # longitude correction
    lc = (standard_meridian-longitude) / 15
    # solar noon (Eq 11.3)
    to = 12 - lc - et
    # Calculate half day length in degrees (Eq 11.6)
    hs = (180 / np.pi) * np.acos(-np.sin((np.pi/180)*(latitude)) * np.sin((np.pi/180)*(sigma)) /\
                                     (np.cos((np.pi/180)*(latitude))*np.cos((np.pi/180)*(sigma))))
    # time of sunrise (Eq 11.7)
    tr = np.round(0.5 + (to - hs/15))
    # time of sunset
    ts = np.round((to + hs/15) - 0.5)
    # hours in day from sunrise to sunset 
    #t = [x for x in range(tr, ts, 1)]
    # get daylength
    #day_length = len(t)
    day_length = ts - tr
    # # hourly zenith angle  (Eq. 11.1)
    # psi = 180/math.pi * math.acos(math.sin((math.pi/180)*(latitude))*math.sin((math.pi/180)*(sigma)) + math.cos((math.pi/180)*(latitude))*\
    #                             math.cos((math.pi/180)*(sigma))*math.cos((math.pi/180)*(15*(t-to))))
    # # optical air mass number (Eq. 11.12)
    # m = pa/(101.3*math.cos((math.pi/180)*(psi)))
    # # hourly flux density of solar radiation perpendicular to solar beam [W/m^2] (Eq 11.11)
    # sph = spo * math.power(tao, m)
    # # hourly flux density of solar radiation on a horizontal surface [W/m^2] (Eq. 11.8)
    # sbh = sph * math.cos((math.pi/180)*(psi))
    # # hourly flux density of diffuse radiation on a surface [W/m^2] (Eq. 11.13)
    # sdh = 0.3 * (1-math.power(tao, m)) * sph * math.cos((math.pi/180)*(psi))
    # # hourly total solar radiation [W/m^2] (Eq. 11.10)
    # sth = sbh + sdh
    # # daily averaged solar radiation (Rgmax) in MJ/m^2/day
    # max_radiation = sum(sth) * 3600/1e6

    # # replace max radiation with higher measured radiation
    # if max_radiation < solar_radiation:
    #     max_radiation = solar_radiation
    # else:
    #     pass

    # return max_radiation, day_length

    return solar_radiation, day_length

def _calculate_cloud_correction(solar_radiation: int, max_radiation: int):

    clouds = 0.2 + 0.8*(solar_radiation/max_radiation)

    return clouds
