import numpy as np
import pandas as pd

def homogenize(data_frame_element):
    """This function deletes unwanted characters from strings."""
    
    if type(data_frame_element) is str:
        data_frame_element = data_frame_element.replace('s','')
        data_frame_element = data_frame_element.replace('*','')
        
    return data_frame_element
    
def encode_weather(data_frame):
    """This function overwrites '1' or '0' for each element in the DailyWeather column."""
    
    # Mask where string values indicate precipitation (NOAA codes)
    subframe = data_frame.loc[data_frame['DailyWeather'].notnull(),'DailyWeather']
    mask = subframe.str.contains('TS|GR|FC|BLPY|BR|DZ|RA|UP')
    data_frame.loc[data_frame['DailyWeather'].notnull(),'DailyWeather'] = np.where(mask, '1', '0')
    
    return data_frame

def clean_data_frame(data_frame):
    """This function processes the NOAA data and formats it for ease of use.""" 
    
    data_frame = encode_weather(data_frame)
    
    # Replace trace amounts of precipitation with 0
    data_frame['DailyPrecipitation'].replace('T', '0.0', inplace=True)
    data_frame['DailyPrecipitation'].replace('Ts', '0.0', inplace=True)
    
    # Replace missing entries with NaN
    data_frame.loc[:,'DailyAverageDryBulbTemperature':'DailySustainedWindSpeed'].replace(' ', np.nan, inplace=True)
    data_frame.loc[:,'DailyAverageDryBulbTemperature':'DailySustainedWindSpeed'].replace('*', np.nan, inplace=True)
    
    # Important to apply homogenize after replacement; .replace() works on data frame elements,
    # but it won't change parts of strings in those elements 
    data_frame = data_frame.applymap(homogenize)
    
    # Remove rows where all entries are NaN
    mask = data_frame.index[data_frame.loc[:,'DailyAverageDryBulbTemperature':].isnull().all(1)]
    data_frame = data_frame.drop(axis=0, index=mask)
    
    # Fill remaining nulls with 0 
    data_frame['DailyWeather'].fillna('0', inplace=True)
    
    # Reformat data frame to have dates as the index
    data_frame['DATE'] = data_frame['DATE'].dt.floor('d')
    data_frame = data_frame.set_index('DATE')
    
    # Convert all entries to numbers
    data_frame = data_frame.apply(pd.to_numeric, errors='coerce')
    
    return data_frame
    
def condense_frame(data_frame):
    """This function takes data from the 3 Atlanta airports and averages them."""
    
    temporary_frame = clean_data_frame(data_frame)
    avg_frame = temporary_frame.groupby(temporary_frame.index).mean()
    
    # The groupby object has 3 values (either all 1 or all 0) for each DailyWeather entry.
    # Since these are the same, keep only the first value in each entry
    avg_frame['DailyWeather'] = temporary_frame['DailyWeather'].groupby(temporary_frame.index).apply(lambda x: x[0])
    
    # Fill NaNs using a rolling mean
    avg_frame.fillna(avg_frame.rolling(window=5, min_periods=1).mean(), inplace=True)
    
    return avg_frame
