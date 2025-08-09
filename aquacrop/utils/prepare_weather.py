#-----------------------------------------------------------------------------------------
# CHANGES IMPLEMENTED BY KADE FLYNN AUGUST 2025
#-----------------------------------------------------------------------------------------
#- Add function prepare_weather_minimum_data() to format weather data that conly contains minimum weather variables of minimum temperature, maximum temperature, incoming shortwave (solar) radiation, precipitation, vapor pressure. As part of this function, reference ET is calculated...
#- Add function to calculate Reference ET according to Allen (1998). Wind speed data is   assumed to be equal to 2 m/s.
#- Add functions to calculate net radiaition from input weather data. Net radiation calculation is based on equations in An Introduction to Environmental Biophysics (Campbell and Norman, 1998). Equations include:
#   - calculate_net_radiation()
#   - calculate_mean_temperature()
#   - calculate_saturated_vapor_pressure()
#   - calculate_clear_sky_emissivity()
#   - calculate_total_radiant_energy_emmitted()
#   - calculate_max_radiation
#   - calculate_cloud_correction()
#-----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np  # required for changes implemented by kade flynn August 2025


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

def prepare_weather_minimum_data(weather_file_path: str, latitude: float, longitude: float, elevation: float):
    """
    Function to read in minimumly available weather data (minimum temperature, maximum temperature, incoming shortwave (solar) radiation, precipitation, vapor pressure) and return a dataframe containing weather data formatted for Aquacrop. These weather variables are commonly available for gridded weather datasets such as Daymet (https://daymet.ornl.gov/). This function uses these data to calculate reference ET.

    Parameters:
    -----------
    weather_file_path: str
        Path to weather file containing minimum data.
    latitude: float
        Latitude of simulation location.
    longitude: float
        Longitude of simulation location.
    elevation: 
        Elevation of simulation location.

    Result:
    -------
    weather_df: pandas.DataFrame
        Data frame containing weather data for Aquacrop

    """
    weather_df = pd.read_csv(weather_file_path, header=0)
    assert len(weather_df.columns) == 7

    # rename the columns
    weather_df.columns = str(
        "Year YearDay Precipitation Solar MaxTemp MinTemp VaporPressure").split()

    # put the weather dates into datetime format
    weather_df['Date_Yj'] = pd.to_datetime(weather_df['Year'].astype(str) + '-' + weather_df['YearDay'].astype(str), format='%Y-%j')
    weather_df['Date'] = weather_df['Date_Yj'].dt.strftime('%Y-%m-%d')
    weather_df = weather_df.drop(["Date_Yj"], axis=1)

    # calculate mean temperature
    weather_df['MeanTemp'] = calcualte_mean_temperature(
        minimum_temperature = weather_df['MinTemp'],
        maximum_temperature = weather_df['MaxTemp'])

    # calculate vapor pressure
    weather_df['SatVaporPressure'], weather_df['Delta'] = calculate_saturated_vapor_pressure(weather_df['MeanTemp'])

    # Calculate Net Radiation
    weather_df['NetRadiation'] = calculate_net_radiation(
        mean_temperature= weather_df['MeanTemp'],
        solar_radiation = weather_df['Solar'],
        vapor_pressure = weather_df['VaporPressure'],
        saturated_vapor_pressure = weather_df['SatVaporPressure'],
        delta = weather_df['Delta'],
        year_day = weather_df['YearDay'],
        latitude = latitude,
        longitude = longitude,
        elevation = elevation)
    
    # Calculate Reference ET
    weather_df['ReferenceET'] = calculate_reference_et(
        net_radiation = weather_df['NetRadiation'],
        vapor_pressure=weather_df['VaporPressure'],
        saturated_vapor_pressure=weather_df['SatVaporPressure'],
        delta=weather_df['Delta'],
        mean_temperature = weather_df['MeanTemp'])

    # set limit on ET0 to avoid divide by zero errors
    weather_df['ReferenceET'] = weather_df['ReferenceET'].clip(lower=0.1)
    weather_df['ReferenceET'] = pd.to_numeric(weather_df['ReferenceET']).round(2)

    columns_to_keep = ['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET', 'Date']

    weather_df = weather_df[columns_to_keep]

    weather_df['Date'] = pd.to_datetime(weather_df['Date'])

    return weather_df


def calculate_reference_et(net_radiation: float, vapor_pressure: float, saturated_vapor_pressure: float, delta: float, mean_temperature: float, wind_speed = 2, ground_heat_flux = 0):

    '''Calculate Reference ET with the FAO Penman-Monteith equation (Equation 6, Allen et al. 1998).

    Parameters:
    -----------
    net_radiation: float
        Net radiation (Mj/m^2). Can be calcualted with the calculate_net_radiation() function
    vapor_pressure : float
        Vapor pressure (kPa)
    saturated_vapor_pressure: float
        Saturated vapor pressure (kPa)
    delta : float
        Slope of the saturated vapor pressure temperature curve
    mean_temperature: float
        Average daily temperature (decrees C)
    wind_speed: float
        Wind speed (m/s). Unless specified, assumed to be 2 m/s based on recommendation of Allen et al. (1998)
    ground_heat_flux: float
        Ground heat Flux. Unless specified, assumed to be 0. This is an appropriate assumption for daily time scales (Allen et al. 1998)
    
    Result:
    -------
    ReferenceET: float
        Reference Evapotranspiration for a grass surface.
    '''
    # Equation 8 Allen et al. (1998)
    psychrometric_constant = 0.665e-3 * vapor_pressure
    
    # vapor pressure deficit
    vapor_pressure_deficit = saturated_vapor_pressure - vapor_pressure


    numerator = 0.408 * delta * (net_radiation - ground_heat_flux) + psychrometric_constant * (900 / (mean_temperature + 273)) * wind_speed * vapor_pressure_deficit

    denominator = delta + psychrometric_constant * (1 + 0.34 * wind_speed)

    reference_et = numerator / denominator

    return reference_et

def calculate_net_radiation(mean_temperature, solar_radiation, vapor_pressure, saturated_vapor_pressure, delta, year_day, latitude, longitude, elevation):
    '''
    Function to calculate the net radiation. General reference is An Introduction to Environmental Biophysics (Campbell and Norman, 1998).

    Parameters:
    -----------
    mean_temperature : float
        Mean daily temperature in degrees celcius
    solar_radiation: float
        Incoming shortwave (solar) radiation in Mj/m^2
    vapor_pressure : float
        Vapor pressure (kPa)
    saturated_vapor_pressure: float
        Saturated vapor pressure (kPa)
    delta : float
        Slope of the saturated vapor pressure temperature curve.
    year_day: int
        Day of year.
    latitude: float
        Latitude of simulation location in degrees.
    longitude: float
        Longitude of simulation location in degrees.
    elevation: float
        Elevation of simulation location (m).
    
    Result:
    -------
    NetRadiation: float
        Net radiation at the simulation location.
    '''

    clear_sky_emissivity = calculate_clear_sky_emissivity(
        mean_temperature = mean_temperature,
        vapor_pressure = vapor_pressure)

    total_radiant_energy_emitted = calculate_total_radiant_energy_emitted(
        mean_temperature = mean_temperature,
        clear_sky_emissivity = clear_sky_emissivity)
    
    max_radiation, day_length = calculate_max_radiation(
        latitude = latitude,
        longitude = longitude,
        elevation = elevation,
        day_of_year = year_day,
        solar_radiation = solar_radiation)
    
    cloud_correction = calculate_cloud_correction(
        solar_radiation = solar_radiation,
        max_radiation = max_radiation)

    albedo = 0.23

    net_radiation = solar_radiation * (1-albedo) + total_radiant_energy_emitted * cloud_correction

    return net_radiation

def calcualte_mean_temperature(minimum_temperature: float, maximum_temperature: float) -> float:
    ''' 
    Calculate the mean temperature

    Parameters:
    ----------
    minimum_temperature: float
    maximum_temperature: float

    Result:
    -------
    mean_temperature: float
    '''
    return (minimum_temperature + maximum_temperature) / 2

def calculate_saturated_vapor_pressure(mean_temperature):
    '''
    Function to calculate the saturated vapor pressure and the slope of the saturated vapor pressure curve using Tetens formula. Equation 3.8 from An Introduction to Environmental Biophysics (Campbell and Norman, 1998).

    Parameters:
    -----------
    mean_temperature: float
        Average daily temperature (degrees C).
    
    Result:
    ------
    Tuple(saturated_vapor_pressure, slope_saturated_vapor_pressure_curve)
    '''
    a = 0.611
    b = 17.502
    c = 240.970

    saturated_vapor_pressure = a * (np.exp(b*mean_temperature/(mean_temperature+c)))

    slope_saturated_vapor_pressure_curve = b*c*saturated_vapor_pressure/\
        ((mean_temperature+c)**2) / 101.3

    return saturated_vapor_pressure, slope_saturated_vapor_pressure_curve

def calculate_clear_sky_emissivity(mean_temperature: int,
                                   vapor_pressure: int):
    '''
    Function to calculate the clear sky emissivity. Equation 10.10 from An Introduction to
    Environmental Biophysics (Campbell and Norman, 1998).

    Paramters:
    ---------
    mean_temperature: float
        Average daily temperature (degrees C).
    vapor_pressure: float
        Vapor pressure (kPa).

    Result:
    -------
    clear_sky_emissivity: float
        The clear sky emissivity. Used to calcualte net radiation
    '''

    clear_sky_emissivity = 1.72 * ((vapor_pressure)/(mean_temperature+273))**(1/7)

    return clear_sky_emissivity

def calculate_total_radiant_energy_emitted(mean_temperature: int,
                                           clear_sky_emissivity: int):
    '''
    Function to calculate the total radiant energy emitted by a blackbody. Equation 10.9 from An Introduction to
    Environmental Biophysics (Campbell and Norman, 1998).

    Paramters:
    ---------
    mean_temperature: float
        Average daily temperature (degrees C).
    clear_sky_eissivity: float
        The clear sky emissivity. Calculated with function calculate_clear_sky_emissivity().

    Result:
    -------
    total_radiant_energy_emitted: float
        total radiant energy emitted by a blackbody.
    '''

    total_radiant_energy_emitted = 0.98 * (clear_sky_emissivity-1) * 5.67e-8 * (mean_temperature+273)**4
    # convert to Mj/m2. Output of Stefan-Boltzmann law is W/m2
    total_radiant_energy_emitted = total_radiant_energy_emitted * 14 * 3600 * 1e-6

    return total_radiant_energy_emitted


def calculate_max_radiation(latitude: int,
                            longitude: int,
                            elevation: int,
                            day_of_year: int,
                            solar_radiation: int):
    '''
    Function to calculate the maximum radiation for a given location. General reference
    is An Introduction to Environmental Biophysics (Campbell and Norman, 1998). pages 168 to 173.
    Parameters:
    -----------
    latitute: float
    longitude: float
    elevation: float
    day_of_year: float
    solar_radiation: float

    Result:
    -------
    max_radiation, day_length

    '''
    standard_meridian = longitude

    # extraterrestrial flux density normal to the solar beam [W/m^2]
    spo = 1360

    # atmoshperic transmittance
    tao = 0.75

    # calculate atmospheric pressure
    pa = 101.3 * np.exp(-elevation/8200)

    # Equation of time (Eq. 11.4)
    f = 279.575 + 0.9856 * day_of_year
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
    t = pd.Series([np.arange(start, stop, 1, dtype=np.float64) for start, stop in zip(tr, ts)])
    day_length = len(t)

    # hourly zenith angle  (Eq. 11.1)
    psi_arrays = [
        180/np.pi * np.arccos(np.sin((np.pi/180)*latitude)*np.sin((np.pi/180)*sigma) + 
                          np.cos((np.pi/180)*latitude)*np.cos((np.pi/180)*sigma)*
                          np.cos((np.pi/180)*(15*(t_array-to))))
        for t_array, to, sigma in zip(t, to, sigma)]
    psi = pd.Series(psi_arrays, dtype=object)

    # optical air mass number (Eq. 11.12)
    m_arrays = [
        pa/(101.3*np.cos((np.pi/180)*(psi_array)))
                for psi_array in psi]
    m = pd.Series(m_arrays, dtype=object)

    # hourly flux density of solar radiation perpendicular to solar beam [W/m^2] (Eq 11.11)
    sph_arrays = [spo * np.power(tao, m_array)
                 for m_array in m]
    sph = pd.Series(sph_arrays, dtype=object)

    # hourly flux density of solar radiation on a horizontal surface [W/m^2] (Eq. 11.8)
    sbh_arrays = [sph_array * np.cos((np.pi/180)*(psi_array))
                 for sph_array, psi_array in zip(sph, psi)]
    sbh = pd.Series(sbh_arrays, dtype=object)

    # hourly flux density of diffuse radiation on a surface [W/m^2] (Eq. 11.13)
    sdh_arrays = [0.3 * (1-np.power(tao, m_array)) * sph_array * np.cos((np.pi/180)*(psi_array))
                  for m_array, sph_array, psi_array in zip(m, sph, psi)]
    sdh = pd.Series(sdh_arrays, dtype=object)

    # hourly total solar radiation [W/m^2] (Eq. 11.10)
    sth = [sbh_array + sdh_array
           for sbh_array, sdh_array in zip(sbh, sdh)]
    
    # daily averaged solar radiation (Rgmax) in MJ/m^2/day
    max_radiation_array = [sum(sth_array) * 3600/1e6
                     for sth_array in sth]
    max_radiation = pd.Series(max_radiation_array, dtype=object)

    # replace max radiation with higher measured radiation
    max_radiation = np.maximum(max_radiation, solar_radiation)

    return max_radiation, day_length

def calculate_cloud_correction(solar_radiation: int, max_radiation: int):

    clouds = 0.2 + 0.8*(solar_radiation/max_radiation)

    return clouds
