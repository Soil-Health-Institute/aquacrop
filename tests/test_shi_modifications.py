
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt
from aquacrop.utils import prepare_weather_minimum_data, get_filepath, prepare_weather

# Prepare weather data
filepath = get_filepath('weather_minimum_lat38.9_lon-83.9_elev291.csv')
weather_data = prepare_weather_minimum_data(weather_file_path = filepath,
                                       latitude = 38.9,
                                       longitude = -83.9,
                                       elevation = 291)

print(weather_data)

print(weather_data['MinTemp'])

# Prepare soil data
soil_baseline = Soil(soil_type='custom', 
                     is_calcareous = False,
                     dz=[0.15]*10,
                     rew = 9)
soil_baseline.add_layer_from_texture_bagnall(thickness=0.15*1,
                                                 Sand=25,
                                                 Clay=25,
                                                 SOC=2.0,
                                                 penetrability=1)
soil_baseline.add_layer_from_texture_bagnall(thickness=0.15*10,
                                                 Sand=25,
                                                 Clay=25,
                                                 SOC=2.0,
                                                 penetrability=1)
soil_health = Soil(soil_type='custom', 
                     is_calcareous = False,
                     dz=[0.15]*10, 
                     rew = 15)
soil_health.add_layer_from_texture_bagnall(thickness=0.15*1,
                                                 Sand=25,
                                                 Clay=25,
                                                 SOC=3.0,
                                                 penetrability=1)
soil_health.add_layer_from_texture_bagnall(thickness=0.15*10,
                                                 Sand=25,
                                                 Clay=25,
                                                 SOC=2.0,
                                                 penetrability=1)
print(soil_baseline)
print(soil_health)
print(soil_baseline.profile)
print(soil_health.profile)

# Prepare crop data
maize = Crop("Maize", planting_date = '04/01')

# Set management
management_baseline = FieldMngt(mulches = False, mulch_pct = 0, f_mulch = 0.5)
management_health= FieldMngt(mulches = True, mulch_pct = 30, f_mulch = 0.5)

# Set initial water content
initial_water_content_baseline = InitialWaterContent(value=["FC"])
initial_water_content_health = InitialWaterContent(value=["FC"])


# Create model
model_baseline = AquaCropModel(
    sim_start_time='2021/04/01',
    sim_end_time='2021/09/01',
    weather_df=weather_data,
    soil = soil_baseline,
    crop = maize,
    field_management=management_baseline,
    initial_water_content = initial_water_content_baseline
)
model_health = AquaCropModel(
    sim_start_time='2021/04/01',
    sim_end_time='2021/09/01',
    weather_df=weather_data,
    soil = soil_health,
    crop = maize,
    field_management=management_health,
    initial_water_content = initial_water_content_health
)

model_baseline.run_model(till_termination=True)
model_health.run_model(till_termination=True)

print(model_baseline._outputs.final_stats)
print(model_health._outputs.final_stats)

baseline_trans = sum(model_baseline._outputs.water_flux['Tr'])
health_trans = sum(model_health._outputs.water_flux['Tr'])
baseline_evap = sum(model_baseline._outputs.water_flux['Es'])
health_evap = sum(model_health._outputs.water_flux['Es'])
print(baseline_trans)
print(health_trans)
print(baseline_evap)
print(health_evap)