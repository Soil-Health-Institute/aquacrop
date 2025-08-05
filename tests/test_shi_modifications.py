from aquacrop import Soil

soil_custom1 = Soil(soil_type = 'custom')
print(soil_custom1)
soil_custom1.add_layer_from_texture(thickness = 0.1, Sand = 15, Clay = 15, OrgMat = 2.5, penetrability = 1)
print(soil_custom1.profile)

soil_custom2 = Soil(soil_type = 'custom', is_calcareous=False)
print(soil_custom2)
soil_custom2.add_layer_from_texture_shi(thickness = 0.1, Sand = 15, Clay = 15, SOC = 2.5*1.724, penetrability = 1)
print(soil_custom2.profile)

import numpy as np
th_fc = 0.412
th_wp = 0.162
th_s = 0.48
lmbda = 1 / ((np.log(1500) - np.log(33)) / (np.log(th_fc) - np.log(th_wp)))
Ksat = (1930 * (th_s - th_fc) ** (3 - lmbda)) * 24
Ksat

2.8*24
(0.1**67)
0.2**67