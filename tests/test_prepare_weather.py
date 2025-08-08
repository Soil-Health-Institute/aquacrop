from aquacrop.utils import get_filepath, prepare_weather, prepare_weather_minimum_data

import pandas as pd

import pydaymet
import pygridmet

# read in aquacrop example data
filepath_1 = get_filepath('champion_climate.txt')
print(filepath_1)
filepath_2 = get_filepath('weather_minimum_lat38.9_lon-83.9_elev291.csv')
print(filepath_2)

# prepare weather data with two methods and compare
weather_data_1 = prepare_weather(filepath_1)
weather_data_2= prepare_weather_minimum_data(filepath_2, 38.9, -83.9, 291)
print(weather_data_1)
print(weather_data_2)


weather_data_2.to_clipboard()

# use py daymet to get weather for given location
coords = (-83.9, 38.9)
crs = 4326
dates = ("1980-01-01", "2023-12-31")
pydaymet_data_2 = pydaymet.get_bycoords(coords, dates, variables=['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl'], crs=crs, time_scale='daily', pet='priestley_taylor')
print(pydaymet_data_2)
pydaymet_data_2.to_clipboard()


pygridmet_data_2 = pygridmet.get_bycoords(coords, dates, crs=crs)
print(pygridmet_data_2)
pygridmet_data_2.to_clipboard()