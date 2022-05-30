import pickle as pkl  
import os 
import pandas as pd 
 
# read csv meta data, dropping the unnamed column by setting index_col = 0 
cwd = os.getcwd()
meta_data_filepath = os.path.join(cwd, "data", "meta_data.csv")
meta_data = pd.read_csv(meta_data_filepath, index_col=0)

# drop the first column that was just meta data and all null values
meta_data = meta_data.drop(['Meta Data'], axis=1) 
# drop rows with all nan values (that were just separators), and drop=True to avoid old indices being added as a new column
meta_data = meta_data.dropna(how='all').reset_index(drop=True) 

# new columns to rename unnamed columns 
keys_list = meta_data.columns
 
new_columns = ['Exam', 'Label']
for q in range(1, 35):
    new_columns.append('q' + str(q))
values_list = new_columns
zip_iterator = zip(keys_list, values_list)
 
a_dictionary = dict(zip_iterator)
# rename columns 
meta_data = meta_data.rename(columns=a_dictionary) # change the labels for the columns of the dataset 
 
pivoted_meta_data = meta_data.pivot(index='Exam', columns = 'Label')
 
meta_dict = pivoted_meta_data.to_dict('index')
 
dictionary_save_path = os.path.join(os.getcwd(), 'data', 'meta_dictionary.p')
with open(dictionary_save_path, 'wb') as fp:
    pkl.dump(meta_dict, fp, protocol=pkl.HIGHEST_PROTOCOL)
