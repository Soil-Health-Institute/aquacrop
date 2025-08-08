import sys

if not "-m" in sys.argv:
    from .prepare_weather import prepare_weather, prepare_weather_minimum_data
    from .data import get_filepath
    from .lars import prepare_lars_weather, select_lars_wdf
