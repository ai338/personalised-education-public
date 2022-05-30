import os 
import pandas as pd 

"""
Load the large dataset in xlsm format and save two separate .csv 
files with data and meta data separately 
"""

# get current folder path 
cwd = os.getcwd()

# get path of the large dataset in xlsm 
large_dataset_filepath = os.path.join(cwd, 'data', 'large_dataset.xlsm')

# read excel file; include both sheets 
xl = pd.ExcelFile(large_dataset_filepath)

# parse meta data and data
data = xl.parse('DATA')
meta_data = xl.parse('Meta Data')

# save meta data and data as csv files for future quicker use 
meta_data_csv_path = os.path.join(cwd, 'data', 'meta_data.csv')
data_csv_path = os.path.join(cwd, 'data', 'data.csv')

meta_data.to_csv(meta_data_csv_path)
data.to_csv(data_csv_path)

