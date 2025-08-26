from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt
from aquacrop.utils import prepare_weather_minimum_data, get_filepath, prepare_weather

import pandas as pd
import matplotlib.pyplot as plt

## create weather
filepath = get_filepath('C:/Users/KadeFlynn/OneDrive - Soil Health Institute/Documents/aquacrop tests/weather_arkansas_lat35.91_long-90.68_elev122.csv')
weather_data = prepare_weather_minimum_data(
        weather_file_path=filepath,
        latitude=35.9,
        longitude=-90.68,
        elevation=122
    )

# create soil
soil = Soil(soil_type ='Loam')

# create crop
crop = Crop(c_name='Maize', planting_date = '04/25')

# create initial water content
inital_water_content = InitialWaterContent(value=['FC'])

# create model
model = AquaCropModel(sim_start_time = '2011/01/01',
                        sim_end_time = '2012/12/31',
                        weather_df=weather_data,
                        soil=soil,
                        crop=crop,
                        initial_water_content=inital_water_content,
                        off_season=True)

# run model
model.run_model(till_termination=True) 

model_results = model.get_simulation_results().head()

print(model_results)

flux = model._outputs.water_flux
storage = model._outputs.water_storage
print(flux.head())
simulation_date = weather_data[(weather_data['Date'] >= pd.to_datetime('2011/01/01')) & (weather_data['Date'] <= pd.to_datetime('2013/01/01'))]['Date']
print(simulation_date)
weather_data['Date']

#create plot of water flux 
fig,ax=plt.subplots(1,1,figsize=(13,8))
# plot results
ax.plot(simulation_date, flux['Tr'], color = 'green')
ax.plot(simulation_date, flux['Es'], color = 'khaki')
# labels
ax.set_xlabel('Day after planting)',fontsize=18)
ax.set_ylabel('Water flux (mm)',fontsize=18)
# legend
#ax.legend(['Transpiration','Soil Evaporation'],fontsize=16)
# show plot
fig.show()  

#create plot of water storage 
fig,ax=plt.subplots(1,1,figsize=(13,8))
# plot results
ax.plot(simulation_date, storage['th1'], color = 'blue')
#ax.plot(simulation_date, flux['Es'], color = 'khaki')
# labels
ax.set_xlabel('Day after planting)',fontsize=18)
ax.set_ylabel('Water flux (m)',fontsize=18)
# legend
ax.legend(['Transpiration','Soil Evaporation'],fontsize=16)
# show plot
fig.show() 


import numpy as np

epotential = 5
mass = 5000
fraction_cover = 1 - np.exp((-42 * 1e-5) * mass)
fraction_cover
fcover = 0.90
mass = np.log(1 - fcover) / -(42 * 1e-5)
mass


mai = 42 * 1e-5 * mass / fcover
mai
em_dssat = epotential *  (1 - np.exp(-0.81 * mai))  * fcover
em_dssat

es_dssat = epotential * np.exp(-0.81 * mai) * fcover + epotential * (1 - fcover)
es_dssat


es_aquacrop = epotential * (1 - 0.5 * (90/100))
es_aquacrop


es_scopel = epotential * np.exp(-0.81 * mai) * fcover + epotential * (1-fcover)
es_scopel

# 30 percent cover aquacrop: 4.25, scopel: 4.07
# 60 percent cover aquacrop: 3.5, scopel: 2.87
# 90 percent cover aquacrop: 2.75, scopel: 1.06 