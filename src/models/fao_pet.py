from omegaconf import DictConfig
import logging
from eto import ETo
import pandas as pd
from functools import cached_property
import os
import math

log = logging.getLogger("models.dCFE")
# 
# https://eto.readthedocs.io/en/latest/package_references.html
# Only few user exsits, possibly swtich to https://github.com/pyet-org/pyet or https://github.com/woodcrafty/PyETo

class FAO_PET():
    
    def __init__(self, cfg: DictConfig, nldas_forcing)-> None:
        """
        :param cfg:
        """
        self.cfg = cfg

        # Get forcing 
        self.nldas_forcing = nldas_forcing
        self.input_forcing = self.prepare_input(nldas_forcing)

        # Get CAMELS basin attributes
        basin_attrs = pd.read_csv(self.cfg.camels_attr_file)
        basin_idx = basin_attrs['basin_id']==self.cfg.data.basin_id
        self.elevation = basin_attrs['elevation'][basin_idx]
        self.zw = 10.0 # basin_attrs['zw'][basin_idx] #TODO: check what is this
        self.lon = basin_attrs['lon'][basin_idx]
        self.lat = basin_attrs['lat'][basin_idx]

        
    def calc_PET(self):
        
        PET = ETo(self.new_df, 'H', z_msl=self.elevation, lat=self.lat, lon=self.lon, TZ_lon=self.lon).eto_fao() 
        PET = PET.fillna(0)
        PET = PET / self.cfg.conversions.m_to_mm  #mm/hr to m/hr  

        return PET
    
    def prepare_input(self, df):

        # Convert the time column to DateTime type
        df['time'] = pd.to_datetime(df['date'])

        # Set the time column as the index
        df.set_index('time', inplace=True)

        # Calculate relative humidity for each row
        df["Relative_Humidity"] = df.apply(self.calculate_relative_humidity, axis=1)

        df["Actual_Vapor_Pressure"] = df.apply(self.calculate_actual_vapor_pressure, axis=1)

        df["day_of_year"] = df.index.dayofyear

        df["utc_hour"] = df.index.tz_localize(None).hour
        
        data = {
            'R_s': df["DSWRF_surface"] * 0.0036,       
            'T_mean': df["TMP_2maboveground"] - 273.15, #C
            'e_a': df["Actual_Vapor_Pressure"] / 1000,  #Kpa
            'RH_mean': df["Relative_Humidity"],
            'date': df.index
        }

        input_forcing = pd.DataFrame(data)
        input_forcing.index = pd.to_datetime(input_forcing.index)
        
        return input_forcing

    
    def calculate_relative_humidity(self, row):
        specific_humidity = row["SPFH_2maboveground"]
        temperature = row["TMP_2maboveground"] - 273.15
        pressure = row["PRES_surface"]/100
        p_sat = 6.112 * math.exp((17.67 * temperature) / (temperature + 243.5))
        relative_humidity = (specific_humidity / (1 - specific_humidity)) * (p_sat / pressure) * 100
        return relative_humidity    
    
    
    def calculate_actual_vapor_pressure(self, row):
        temperature = row["TMP_2maboveground"] - 273.15  # Temperature in Celsius
        specific_humidity = row["SPFH_2maboveground"]  # Specific humidity
        pressure = row["PRES_surface"]  # Atmospheric pressure in pascals

        actual_vapor_pressure = specific_humidity * pressure / (0.622 + 0.378 * specific_humidity)
        return actual_vapor_pressure    

