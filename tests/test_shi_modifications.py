
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt
from aquacrop.utils import prepare_weather_minimum_data, get_filepath, prepare_weather

import matplotlib.pyplot as plt
from typing import List, Dict
import pandas as pd
import numpy as np
from itertools import product
import warnings

## Prepare weather data
filepath = get_filepath('weather_minimum_lat38.9_lon-83.9_elev291.csv')
weather_data = prepare_weather_minimum_data(weather_file_path = filepath,
                                       latitude = 38.9,
                                       longitude = -83.9,
                                       elevation = 291)

## Prepare Soil data

def create_custom_soil_profile(
    sand_list,
    clay_list,
    soc_list=None,
    layer_thickness_list=None,
    penetrability_list=None,
    is_calcareous=False,
    rew=9,
    z_res=-999
):
    """
    Create a soil profile with variable sand and clay content for each layer.
    
    Parameters:
    -----------
    sand_list : list
        List of sand percentages for each soil layer (e.g., [25, 30, 35, 40])
    clay_list : list  
        List of clay percentages for each soil layer (e.g., [25, 20, 15, 10])
    layer_thickness_list : list, optional
        List of layer thicknesses in meters. If None, uses 0.15m for all layers
    soc_list : list, optional
        List of soil organic carbon percentages for each layer. If None, uses 2.0 for all
    penetrability_list : list, optional
        List of penetrability values (0-100) for each layer. If None, uses 100 for all
    is_calcareous : bool
        Whether soil is calcareous
    rew : float
        Readily evaporable water parameter
    z_res : float
        Residual soil layer thickness
        
    Returns:
    --------
    Soil object with layers created according to input lists
    
    Raises:
    -------
    ValueError: If sand_list and clay_list have different lengths
    """
    
    # Validate inputs
    if len(sand_list) != len(clay_list):
        raise ValueError(f"sand_list length ({len(sand_list)}) must equal clay_list length ({len(clay_list)})")
    
    n_layers = len(sand_list)
    
    # Set default values if not provided
    if layer_thickness_list is None:
        layer_thickness_list = [0.15] * n_layers
    elif len(layer_thickness_list) != n_layers:
        raise ValueError(f"layer_thickness_list length must equal sand/clay list length ({n_layers})")
    
    if soc_list is None:
        soc_list = [2.0] * n_layers
    elif len(soc_list) != n_layers:
        raise ValueError(f"soc_list length must equal sand/clay list length ({n_layers})")
    
    if penetrability_list is None:
        penetrability_list = [100] * n_layers
    elif len(penetrability_list) != n_layers:
        raise ValueError(f"penetrability_list length must equal sand/clay list length ({n_layers})")
    
    
    # Create soil object
    soil = Soil(
        soil_type='custom',
        is_calcareous=is_calcareous,
        dz=layer_thickness_list,
        rew=rew,
        z_res=z_res
    )
    
    # Add each layer with specified properties
    for i in range(n_layers):
        soil.add_layer_from_texture_bagnall(
            thickness=layer_thickness_list[i] + 0.01,
            Sand=sand_list[i],
            Clay=clay_list[i],
            SOC=soc_list[i],
            penetrability=penetrability_list[i]
        )
        
        # print(f"Added Layer {i+1}: "
        #       f"Thickness={layer_thickness_list[i]}m, "
        #       f"Sand={sand_list[i]}%, "
        #       f"Clay={clay_list[i]}%, "
        #       f"Silt={100-sand_list[i]-clay_list[i]}%, "
        #       f"SOC={soc_list[i]}%, "
        #       f"Penetrability={penetrability_list[i]}")

    # overwrite tau calculation
    soil.profile['tau'] = 1

    # calulate REW as in DRC
    surface_fc = soil.profile['th_fc'][0]
    surface_dry = soil.profile['th_dry'][0]
    soil.rew = 1000 * (surface_fc - surface_dry) * 0.04

    
    return soil


def analyze_soil_health_impacts(weather_data: pd.DataFrame,
                                depth_increments: List[float],
                                sand_profile: List[float],
                                clay_profile: List[float],
                                soc_profile: List[float],
                                crop: str,
                                planting_date: str,
                                year: int,
                                soc_increase: float,
                                residue_cover: int):
    '''
        Analyze the impact of soil health improvements on transpiration, evaporation, and soil water.
    
    Parameters:
    -----------
    weather_data: pd.DataFrame
        weather data formatted for Aquacrop with function prepare_weather_minimum_data()
    depth_increments: List[float]
        List of soil compartment thickness in meters
    sand_profile : List[float]
        Sant content (percent) for each soil compartment
    clay_profile : List[float]  
        Clay content (percent) for each soil compartment)
    soc_profile : List[float]
        Soc conent (percent) for each soil compartment
    crop: str
        crop type
    planting date: str
        Date of planting mm/dd
    year : int
        Year for simulation
    soc_increase : float
        Amount to increase SOC(%) in the soil health managemnt scenario
    residue_cover : int
        Residue (mulch) cover in the soil health manageent scenario (%)

        
    Returns:
    --------
    dict containing:
        - transpiration_difference: Difference in total transpiration (health - baseline)
        - evaporation_difference: Difference in total evaporation (health - baseline) 
        - baseline_water_df: DataFrame of soil water content over time for baseline
        - health_water_df: DataFrame of soil water content over time for health scenario
        - baseline_flux_df: DataFrame of water fluxes over time for baseline
        - health_flux_df: DataFrame of water fluxes over time for health scenario
        - summary_stats: Dictionary of final simulation statistics
    '''

    ## create soil profiles
    # Get SOC profile for soil health soil
    soc_profile_health = soc_profile.copy()
    soc_profile_health[0] = soc_profile[0] + soc_increase

    # create soil objects 
    soil_base = create_custom_soil_profile(sand_list = sand_profile,
                                           clay_list = clay_profile,
                                           soc_list = soc_profile,
                                           layer_thickness_list= depth_increments,
                                           penetrability_list = [100]*len(sand_profile),
                                           is_calcareous=False,
                                           rew=9,
                                           z_res=-999)
    soil_health = create_custom_soil_profile(sand_list = sand_profile,
                                           clay_list = clay_profile,
                                           soc_list = soc_profile_health,
                                           layer_thickness_list= depth_increments,
                                           penetrability_list = [100]*len(sand_profile),
                                           is_calcareous=False,
                                           rew=9,
                                           z_res=-999)
    
    # create crop object
    crop = Crop(c_name = crop, planting_date = planting_date)
    crop.Zmax = 1.5

    # create management objects
    management_base = FieldMngt(mulches = False)
    management_health = FieldMngt(mulches = True, mulch_pct = residue_cover, f_mulch = 0.5)

    # set start date and end date
    sim_start = f'{year}/{planting_date}'
    start_date = pd.to_datetime(sim_start)
    end_date = start_date + pd.Timedelta(days=200)
    sim_end = end_date.strftime('%Y/%m/%d')
    
    # set initial water content
    layers = list(range(1, 11))
    initial_water_content = InitialWaterContent(wc_type = 'Prop',
                                               method = 'Layer',
                                               depth_layer = layers,
                                               value = ['FC'] * len(layers))

    # create model objects
    model_base = AquaCropModel(sim_start_time = sim_start,
                               sim_end_time = sim_end,
                               weather_df = weather_data,
                               soil = soil_base,
                               crop = crop,
                               initial_water_content = initial_water_content,
                               field_management = management_base)
    model_health = AquaCropModel(sim_start_time = sim_start,
                               sim_end_time =sim_end,
                               weather_df = weather_data,
                               soil = soil_health,
                               crop = crop,
                               initial_water_content = initial_water_content,
                               field_management = management_health)
    
    # print input data
    print(f"Simulation start date is {sim_start}"
          f" Simulation end date is {sim_end}"
          f" Crop planting date 9s {crop.planting_date}")

    # run models
    model_base.run_model(till_termination=True)
    model_health.run_model(till_termination=True)

    # extract days to harvest 
    days_to_harvest_base = model_base._outputs.final_stats['Harvest Date (Step)'][0] + 1
    days_to_harvest_health = model_health._outputs.final_stats['Harvest Date (Step)'][0] + 1

    sim_length = min(days_to_harvest_base, days_to_harvest_health)

    # Calculate transpiration and evaporation totals
    base_trans = model_base._outputs.water_flux['Tr'][:sim_length].sum()
    health_trans = model_health._outputs.water_flux['Tr'][:sim_length].sum()
    base_evap = model_base._outputs.water_flux['Es'][:sim_length].sum()
    health_evap = model_health._outputs.water_flux['Es'][:sim_length].sum()
    
    # Get output DataFrames
    base_flux_df = model_base._outputs.water_flux[:sim_length].copy()
    health_flux_df = model_health._outputs.water_flux[:sim_length].copy()
    base_water_df = model_base._outputs.water_storage[:sim_length].copy()
    health_water_df = model_health._outputs.water_storage[:sim_length].copy()
    base_crop_growth_df = model_base._outputs.crop_growth[:sim_length].copy()
    health_crop_growth_df = model_health._outputs.crop_growth[:sim_length].copy()
    
    # Calculate total water content and plant available water
    layer_thickness_mm = [x * 1000 for x in depth_increments]  # Convert to mm
    
    def calculate_water_totals(water_df, soil_profile, n_layers, layer_thickness_mm):
        """Calculate total water content for all layers"""
        water_df = water_df.copy()
        
        # Calculate water content in mm for each layer
        for i in range(1, n_layers+1):
            water_df[f'th{i}_mm'] = water_df[f'th{i}'] * layer_thickness_mm[i-1]
        # Calculate wilting point in mm for each layer
        soil_profile[f'th_wp_mm'] = soil_profile[f'th_wp'] * layer_thickness_mm[i-1]
        th_wp_total_mm = soil_profile['th_wp_mm'].sum(axis=0)
        # Calculate total water content
        th_columns = [f'th{i}_mm' for i in range(1, n_layers+1)]
        water_df['th_total_mm'] = water_df[th_columns].sum(axis=1)
        water_df['paw_total_mm'] = water_df['th_total_mm'] - th_wp_total_mm
        
        return water_df
    
    base_water_df = calculate_water_totals(base_water_df, soil_base.profile, len(layer_thickness_mm), layer_thickness_mm)
    health_water_df = calculate_water_totals(health_water_df, soil_health.profile, len(layer_thickness_mm), layer_thickness_mm)
    
    # Compile results
    results = {
        'transpiration_difference': health_trans - base_trans,
        'evaporation_difference': health_evap - base_evap,
        'base_transpiration_total': base_trans,
        'health_transpiration_total': health_trans,
        'base_evaporation_total': base_evap,
        'health_evaporation_total': health_evap,
        'base_water_df': base_water_df,
        'health_water_df': health_water_df,
        'base_flux_df': base_flux_df,
        'health_flux_df': health_flux_df,
        'base_crop_growth_df': base_crop_growth_df,
        'health_crop_growth_df': health_crop_growth_df,
        'summary_stats': {
            'base': model_base._outputs.final_stats,
            'health': model_health._outputs.final_stats
        }
    }
    soils = {
        'soil_base': soil_base,
        'soil_health': soil_health
    }
    
    return results, soils
    
# results, soils = analyze_soil_health_impacts(weather_data = weather_data,
#                             depth_increments = [0.15]*10,
#                             sand_profile = [65]*10,
#                             clay_profile = [15]*10,
#                             soc_profile = [1]*10,
#                             crop = 'Maize',
#                             planting_date = '04/15',
#                             year =2007,
#                             soc_increase = 1,
#                             residue_cover = 60)

# print(results['base_flux_df'])
# results['base_flux_df'].to_csv("C:/Users/KadeFlynn/OneDrive - Soil Health Institute/Documents/aquacrop tests/aquacrop_base_flux_2007_sl_maxtau.csv", index=False)

# # create plot of paw in soil profile
# fig,ax=plt.subplots(1,1,figsize=(13,8))
# # plot results
# ax.bar(results['health_water_df']['dap'], results['health_water_df']['paw_total_mm'] / 25.4, color = 'lightskyblue')
# ax.bar(results['base_water_df']['dap'], results['base_water_df']['paw_total_mm'] / 25.4, color = 'khaki')
# # labels
# ax.set_xlabel('Day after planting)',fontsize=18)
# ax.set_ylabel('Plant Available Water (in)',fontsize=18)
# #limits
# ax.set_ylim(min(results['base_water_df']['paw_total_mm'] / 25.4), max(results['health_water_df']['paw_total_mm'] / 25.4))
# # show plot
# fig.show()  

# # create plot of root growth
# fig,ax=plt.subplots(1,1,figsize=(13,8))
# # plot results
# #ax.scatter(results['health_crop_growth_df']['dap'], results['health_crop_growth_df']['z_root'], color = 'lightskyblue')
# ax.scatter(results['base_water_df']['dap'], results['base_water_df']['z_root'], color = 'khaki')
# # labels
# ax.set_xlabel('Day after planting)',fontsize=18)
# ax.set_ylabel('Root Depth (m)',fontsize=18)
# # show plot
# fig.show()  

def run_multiyear_analysis(
    weather_data: pd.DataFrame,
    years: List[int],
    soil_scenarios: List[Dict],
    soil_health_scenarios: List[Dict],
    crop: str = "Soybean",
    planting_date: str = "04/20",
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    Run multi-year analysis across multiple soil scenarios and soil health treatments.
    
    Parameters:
    -----------
    weather_data : pd.DataFrame
        Weather data formatted for AquaCrop
    years : List[int]
        List of years to simulate (e.g., [1988, 1989, 1990])
    soil_scenarios : List[Dict]
        List of soil scenario dictionaries, each containing:
        - 'name': Scenario name
        - 'depth_increments': List of layer thicknesses
        - 'sand_profile': List of sand percentages
        - 'clay_profile': List of clay percentages  
        - 'soc_profile': List of SOC percentages
    soil_health_scenarios : List[Dict]
        List of soil health treatment dictionaries, each containing:
        - 'name': Treatment name
        - 'soc_increase': SOC increase amount (%)
        - 'residue_cover': Residue cover percentage
    crop : str
        Crop type
    planting_date : str
        Planting date in 'MM/DD' format
    output_csv_path : str, optional
        Path to save results CSV file
        
    Returns:
    --------
    pd.DataFrame with columns:
        - year, soil_scenario, health_scenario, transpiration_diff, evaporation_diff, 
          base_transpiration, health_transpiration, base_evaporation, health_evaporation
    """
    
    results_list = []
    total_runs = len(years) * len(soil_scenarios) * len(soil_health_scenarios)
    current_run = 0
    
    print(f"Starting multi-year analysis: {total_runs} total simulations")
    print(f"Years: {len(years)}, Soil scenarios: {len(soil_scenarios)}, Health scenarios: {len(soil_health_scenarios)}")
    
    # Iterate through all combinations
    for year, soil_scenario, health_scenario in product(years, soil_scenarios, soil_health_scenarios):
        current_run += 1
        
        try:
            print(f"Run {current_run}/{total_runs}: Year {year}, "
                  f"Soil: {soil_scenario['name']}, Health: {health_scenario['name']}")
            
            # Run single year analysis
            results, soils = analyze_soil_health_impacts(
                weather_data=weather_data,
                depth_increments=soil_scenario['depth_increments'],
                sand_profile=soil_scenario['sand_profile'],
                clay_profile=soil_scenario['clay_profile'],
                soc_profile=soil_scenario['soc_profile'],
                crop=crop,
                planting_date=planting_date,
                year=year,
                soc_increase=health_scenario['soc_increase'],
                residue_cover=health_scenario['residue_cover']
            )
            
            # Store results
            result_row = {
                'year': year,
                'soil_scenario': soil_scenario['name'],
                'health_scenario': health_scenario['name'],
                'soc_increase': health_scenario['soc_increase'],
                'residue_cover': health_scenario['residue_cover'],
                'transpiration_diff': results['transpiration_difference'],
                'evaporation_diff': results['evaporation_difference'],
                'base_transpiration': results['base_transpiration_total'],
                'health_transpiration': results['health_transpiration_total'],
                'base_evaporation': results['base_evaporation_total'],
                'health_evaporation': results['health_evaporation_total'],
                'transpiration_pct_change': ((results['health_transpiration_total'] - results['base_transpiration_total']) / results['base_transpiration_total']) * 100,
                'evaporation_pct_change': ((results['health_evaporation_total'] - results['base_evaporation_total']) / results['base_evaporation_total']) * 100
            }
            
            # Add soil texture info
            result_row['avg_sand'] = np.mean(soil_scenario['sand_profile'])
            result_row['avg_clay'] = np.mean(soil_scenario['clay_profile'])
            result_row['surface_sand'] = soil_scenario['sand_profile'][0]
            result_row['surface_clay'] = soil_scenario['clay_profile'][0]
            
            # Add crop performance metrics if available
            if 'summary_stats' in results:
                base_stats = results['summary_stats']['base']
                health_stats = results['summary_stats']['health']
                
                result_row['base_yield'] = base_stats.get('Yield (tonne/ha)', [np.nan])[0]
                result_row['health_yield'] = health_stats.get('Yield (tonne/ha)', [np.nan])[0]
                result_row['yield_diff'] = result_row['health_yield'] - result_row['base_yield']
                
                if result_row['base_yield'] > 0:
                    result_row['yield_pct_change'] = (result_row['yield_diff'] / result_row['base_yield']) * 100
                else:
                    result_row['yield_pct_change'] = np.nan
            
            results_list.append(result_row)
            
        except Exception as e:
            print(f"Error in run {current_run}: {str(e)}")
            # Store error row with NaN values
            error_row = {
                'year': year,
                'soil_scenario': soil_scenario['name'],
                'health_scenario': health_scenario['name'],
                'soc_increase': health_scenario['soc_increase'],
                'residue_cover': health_scenario['residue_cover'],
                'error': str(e)
            }
            results_list.append(error_row)
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    print(f"\nAnalysis complete! Generated {len(results_df)} result rows")
    
    # Save to CSV if path provided
    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to: {output_csv_path}")
    
    return results_df


def create_soil_texture_scenarios() -> List[Dict]:
    """
    Create predefined soil texture scenarios representing different soil types.
    
    Returns:
    --------
    List of soil scenario dictionaries
    """
    
    scenarios = []
    
    # Sandy Loam
    scenarios.append({
        'name': 'Sandy_Loam',
        'depth_increments': [0.15] * 10,
        'sand_profile': [65]*10,
        'clay_profile': [15]*10,
        'soc_profile': [1]*10
    })
    
    # Loam
    scenarios.append({
        'name': 'Loam',
        'depth_increments': [0.15] * 10,
        'sand_profile': [40]*10,
        'clay_profile': [25]*10,
        'soc_profile': [1]*10
    })
    
    # Silty Clay
    scenarios.append({
        'name': 'Silty Clay',
        'depth_increments': [0.15] * 10,
        'sand_profile': [10]*10,
        'clay_profile': [45]*10,
        'soc_profile': [1]*10
    })
    
    return scenarios


def create_soil_health_scenarios() -> List[Dict]:
    """
    Create soil health treatment scenarios.
    
    Returns:
    --------
    List of soil health scenario dictionaries
    """
    
    scenarios = []
    
    # Low improvement
    scenarios.append({
        'name': 'Low_Health',
        'soc_increase': 0.5,
        'residue_cover': 30
    })
    
    # Medium improvement  
    scenarios.append({
        'name': 'Medium_Health',
        'soc_increase': 1.0,
        'residue_cover': 60
    })
    
    # High improvement
    scenarios.append({
        'name': 'High_Health',
        'soc_increase': 2,
        'residue_cover': 90
    })
    
    return scenarios


# Example usage
if __name__ == "__main__":
    # Load weather data (using your existing code)
    filepath = get_filepath('weather_minimum_lat38.9_lon-83.9_elev291.csv')
    weather_data = prepare_weather_minimum_data(
        weather_file_path=filepath,
        latitude=38.9,
        longitude=-83.9,
        elevation=291
    )
    
    # Define analysis parameters
    #years = [1988, 1989, 1990, 1991, 1992]  # Multiple years
    years = list(range(1988, 2019))
    soil_scenarios = create_soil_texture_scenarios()  # Different soil types
    health_scenarios = create_soil_health_scenarios()  # Different health treatments
    
    # Run comprehensive analysis
    results_df = run_multiyear_analysis(
        weather_data=weather_data,
        years=years,
        soil_scenarios=soil_scenarios,
        soil_health_scenarios=health_scenarios,
        crop="Maize",
        planting_date="04/15",
        output_csv_path="C:/Users/KadeFlynn/OneDrive - Soil Health Institute/Documents/aquacrop tests/soil_health_analysis_results_maxtau_rewcalc.csv"
    )
    
    # Display summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total simulations: {len(results_df)}")
    print(f"Average transpiration difference: {results_df['transpiration_diff'].mean():.2f} mm")
    print(f"Average evaporation difference: {results_df['evaporation_diff'].mean():.2f} mm")
    
    # Group by scenarios
    print("\n=== BY SOIL HEALTH SCENARIO ===")
    health_summary = results_df.groupby('health_scenario').agg({
        'transpiration_diff': ['mean', 'std'],
        'evaporation_diff': ['mean', 'std'],
        'yield_diff': ['mean', 'std']
    }).round(2)
    print(health_summary)
    
    print("\n=== BY SOIL TEXTURE ===")
    soil_summary = results_df.groupby('soil_scenario').agg({
        'transpiration_diff': ['mean', 'std'],
        'evaporation_diff': ['mean', 'std'],
        'yield_diff': ['mean', 'std']
    }).round(2)
    print(soil_summary)
    
    # Show sample of results
    print(f"\n=== SAMPLE RESULTS ===")
    print(results_df[['year', 'soil_scenario', 'health_scenario', 
                     'transpiration_diff', 'evaporation_diff', 'yield_diff']].head(10))
    


print(results_df.columns)
results_df['evaporation_pct_change']

fig,ax=plt.subplots(1,1,figsize=(13,8))
# plot results
ax.scatter(results_df['evaporation_pct_change'], results_df['transpiration_pct_change'])
# labels
ax.set_xlabel('Evaporation Change (%)',fontsize=18)
ax.set_ylabel('Transpiration Change (%)',fontsize=18)
#ax.set_xlim(-100,100)
#ax.set_ylim(-100, 100)
# show plot
fig.show() 

fig,ax=plt.subplots(1,1,figsize=(13,8))
# plot results
ax.scatter(results_df['evaporation_pct_change'], results_df['transpiration_pct_change'])
# labels
ax.set_xlabel('Evaporation Change (%)',fontsize=18)
ax.set_ylabel('Transpiration Change (%)',fontsize=18)
#ax.set_xlim(-100,100)
#ax.set_ylim(-100, 100)
# show plot
fig.show() 

