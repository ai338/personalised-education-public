import pandas as pd 
from functions_test_manipulations import get_dictionary, get_merged_test_and_dict, get_large_dataset
import torch 

data, meta_dict = get_large_dataset()

# all_test_names = ['9to1_2017_GCSE_1H', '9to1_2017_GCSE_2H','9to1_2017_GCSE_3H',
#  '9to1_Edexcel_GCSE_Nov2018_1H','9to1_Edexcel_GCSE_Nov2018_2H',
#  '9to1_Edexcel_GCSE_Nov2018_3H','9to1_Edexcel_Nov2017_GCSE_2H',
#  '9to1_Edexcel_Nov2017_GCSE_3H','9to1_GCSE_Edxcel_Summer2018_1H',
#  '9to1_GCSE_Edxcel_Summer2018_2H','9to1_GCSE_Edxcel_Summer2018_3H',
#  '9to1_GCSE_Edxcel_Summer2019_2H','9to1_GCSE_Edxcel_Summer2019_3H']

all_test_names = ['9to1_2017_GCSE_1H', '9to1_2017_GCSE_2H', 
'9to1_2017_GCSE_3H', '9to1_Edexcel_GCSE_Nov2018_1H', '9to1_Edexcel_GCSE_Nov2018_2H', 
'9to1_Edexcel_GCSE_Nov2018_3H', '9to1_Edexcel_Nov2017_GCSE_2H', '9to1_Edexcel_Nov2017_GCSE_3H', 
'9to1_GCSE_Edxcel_Summer2018_1H ', '9to1_GCSE_Edxcel_Summer2018_2H ', '9to1_GCSE_Edxcel_Summer2018_3H ', 
'9to1_GCSE_Edxcel_Summer2019_2H', '9to1_GCSE_Edxcel_Summer2019_3H']

merge_exam_names = ['9to1_2017_GCSE_1H', '9to1_2017_GCSE_2H', 
'9to1_2017_GCSE_3H', '9to1_Edexcel_GCSE_Nov2018_1H', '9to1_Edexcel_GCSE_Nov2018_2H', 
'9to1_Edexcel_GCSE_Nov2018_3H', '9to1_Edexcel_Nov2017_GCSE_2H', '9to1_Edexcel_Nov2017_GCSE_3H']
df_merged, dict_merged, IDs = get_merged_test_and_dict(merge_exam_names)


print(df_merged.shape)
# all_test_names = ['9to1_2017_GCSE_1H', '9to1_2017_GCSE_2H', 
# '9to1_2017_GCSE_3H', '9to1_Edexcel_GCSE_Nov2018_1H', '9to1_Edexcel_GCSE_Nov2018_2H', 
# '9to1_Edexcel_GCSE_Nov2018_3H', '9to1_Edexcel_Nov2017_GCSE_2H', '9to1_Edexcel_Nov2017_GCSE_3H', 