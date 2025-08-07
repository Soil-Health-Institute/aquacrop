from aquacrop import Soil

soil_custom1 = Soil(soil_type = 'custom')
print(soil_custom1)
soil_custom1.add_layer_from_texture(thickness = 0.1, Sand = 15, Clay = 15, OrgMat = 10, penetrability = 1)
print(soil_custom1.profile)

soil_custom2 = Soil(soil_type = 'custom', is_calcareous=False)
print(soil_custom2)
soil_custom2.add_layer_from_texture_bagnall(thickness = 0.1, Sand = 15, Clay = 15, SOC = 10*1.724, penetrability = 1)
print(soil_custom2.profile)
