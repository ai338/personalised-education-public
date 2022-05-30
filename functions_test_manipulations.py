import pandas as pd 
from functools import reduce
import os 
import pickle as pkl 



def get_test(data, test_name):
    """
    From the large dataset, select only the test with the eg. test_name = 9to1_2017_GCSE_1H
    """
    test = data.loc[data['TestName'] == test_name]
    # remove all columns that have all nan values, tests have different number of questions 
    test = test.dropna(axis=1, how = 'all')
    return test  

def merge_tests(data, test_names, argument):
    tests = []

    for name in test_names:
        tests.append(get_test(data, name))
    for ind, test in enumerate(tests):
        if 'TestName' in test.columns:
            tests[ind] = test.drop(columns = ['TestName'])
    
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=[argument],
                                            how='inner'), tests)
    # number of questions in the merged dataset 
    q = len(df_merged.columns) - 1 

    # rename the columns of the merged dataset 
    new_columns = ['q' + str(i) for i in range(1, q+1)]
    df_merged.columns = [df_merged.columns[0]] + new_columns
    
    return df_merged


def get_dictionary(meta_dict, test_name):
    """
    From meta dictionary extracts the dictionary for a single test 
    """
    test_dict = {}
    for i in range(1, 35):
        q = "q" + str(i)
        test_dict[q] = {}

    for key in meta_dict[test_name].keys():
        if not isinstance(meta_dict[test_name][key], float) and key[1] != 'Name':
            test_dict[key[0]][key[1]] = meta_dict[test_name][key]
            
    # remove keys that have empty values from the dictionary 
    test_dict = {k: v for k, v in test_dict.items() if v}

    # Max will be used in ordinal regression so convert to the int -- TODO might need to go to float depending on the code for the ordinal regress. 
    for key in test_dict.keys():
        test_dict[key]['Max'] = int(test_dict[key]['Max'])


    return test_dict 


def merge_dictionaries(meta_dict, test_names):
    dicts = []
    for name in test_names:
        dicts.append(get_dictionary(meta_dict, name))
    # number of questions in the first set 
    n = len(dicts[0].keys())

    for i in range(1, len(dicts)):
        new_dict = {}
        for old_key in dicts[i].keys():
            new_key = 'q' + str(int(old_key[1:]) + n) 
            new_dict[new_key] = dicts[i][old_key]
        n += len(new_dict.keys())
        dicts[i] = new_dict
    
    merged_dictionaries = dicts[0]

    for dictionary in dicts:
        merged_dictionaries = merged_dictionaries | dictionary 
    
    # Max will be used in ordinal regression so convert to the int -- TODO might need to go to float depending on the code for the ordinal regress. 
    for key in merged_dictionaries.keys():
        merged_dictionaries[key]['Max'] = int(merged_dictionaries[key]['Max'])

    return merged_dictionaries 
    

def get_merged_test_and_dict(test_names):
    # load meta data dictionary 
    dict_path = os.path.join(os.getcwd(), 'data', 'meta_dictionary.p')
    file = open(dict_path, 'rb')
    meta_dict = pkl.load(file)
    file.close()

    cwd = os.getcwd()
    data_filepath = os.path.join(cwd, "data", "data.csv")
    data = pd.read_csv(data_filepath, index_col=0)


    # drop rows with all nan values 
    data = data.dropna(how='all').reset_index(drop=True)

    # set the first row of the dataset to be the column values, and drop the first row of the dataset 
    data.columns = data.iloc[0]
    data.drop(data.index[0], inplace = True)


    df_merged = merge_tests(data, test_names, 'AnonymousStudentID')
    dict_merged = merge_dictionaries(meta_dict, test_names)

    # drop column in test_m that has just studentIDs 
    if 'AnonymousStudentID' in df_merged.columns:
        IDs = df_merged['AnonymousStudentID']
        df_merged.drop(columns=['AnonymousStudentID'], inplace=True)
        
    for col in df_merged:
        max_score = dict_merged[col]['Max']
        df_merged[col].loc[df_merged[col] > max_score] = max_score
        # scores that are less than 0 put to 0 
        df_merged[col].loc[df_merged[col] < 0] = 0
    
    
    return df_merged, dict_merged, IDs 

def get_large_dataset():
    # load meta data dictionary 
    dict_path = os.path.join(os.getcwd(), 'data', 'meta_dictionary.p')
    file = open(dict_path, 'rb')
    meta_dict = pkl.load(file)
    file.close()

    cwd = os.getcwd()
    data_filepath = os.path.join(cwd, "data", "data.csv")
    data = pd.read_csv(data_filepath, index_col=0)

    # drop rows with all nan values 
    data = data.dropna(how='all').reset_index(drop=True)

    # set the first row of the dataset to be the column values, and drop the first row of the dataset 
    data.columns = data.iloc[0]
    data.drop(data.index[0], inplace = True)

    return data, meta_dict 
