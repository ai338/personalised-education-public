import pandas as pd
import numpy as np
import torch

import os 
import csv
import xlrd
import pickle as pkl 

def import_data(datafile):
    data = pd.read_csv(datafile)
    questions_info_df = data.head(5)
    questions = list(questions_info_df.columns)[2:-2]
    questions_info = {}
    for question in questions:
        sup_dict = {'Actual': questions_info_df[question][0],
                    'Max': float(questions_info_df[question][1]),
                    'Topics': questions_info_df[question][2],
                    'Difficulty': questions_info_df[question][3],
                    'qtype': questions_info_df[question][4]}

        questions_info[question] = sup_dict

    # keep only real students
    data = data.loc[data['StudentID'].str[0:4].isin(['Real'])]

    # remove duplicates
    data.drop_duplicates(subset=['StudentID'], inplace=True)
    assert (not data['StudentID'].duplicated().any())

    # drop all columns that we won't use for now
    data = data.drop(columns=['Name', 'Formal', 'date', 'StudentID'])

    # let question scores be float values instead of string values
    data = data.astype(float)

    # if someone score above max score for that question, let the value of the score be maximum (not above)
    for col in data:
        max_score = questions_info[col]['Max']
        data[col].loc[data[col] > max_score] = max_score

    return data, questions_info


def binarise_using_average(data):

    data_binarised = data.copy()

    for col in data:
        mean_score = data[col].mean()
        data_binarised[col] = (data[col] >= mean_score).astype(float)
    data_binarised.reset_index(inplace=True, drop=True)
    return data_binarised


def binarise_using_max(data, questions_info):

    data_binarised = data.copy()

    for col in data:
        max_score = questions_info[col]['Max']
        data_binarised[col] = (data[col] >= max_score / 2).astype(float)
    data_binarised.reset_index(inplace=True, drop=True)
    return data_binarised


def shuffle_data(raw_data, shuffle_seed=1000):
    df = raw_data.sample(frac=1, axis=1, random_state=np.random.RandomState(shuffle_seed))
    return df


def save_dataframe(dataframe, save_location):
    dataframe.to_csv(save_location)


def load_dataframe(dataframe_location):
    dataframe = pd.read_csv(dataframe_location)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
    return dataframe


def convert_df_to_tensor(data):
    torch_tensor = torch.tensor(data.values)
    return torch_tensor


def clean_exam(raw_exam_data):
    questions_info_df = raw_exam_data.head(5)
    questions = list(questions_info_df.columns)[1:]
    questions_info = {}
    for question in questions:
        sup_dict = {'Actual': questions_info_df[question][0],
                    'Max': float(questions_info_df[question][1]),
                    'Topics': questions_info_df[question][2],
                    'Difficulty': questions_info_df[question][3],
                    'qtype': questions_info_df[question][4]}

        questions_info[question] = sup_dict
    data = raw_exam_data[6:]
    data = data.dropna(subset=list(questions_info_df.columns)[0:1])
    data.reset_index(inplace=True)
    data = data.astype(float)

    data = data[questions]
    return data, questions_info


def remove_name_index_new_data(df):
    # remove name and index for now as we are just interested in replicating previous results
    questions = df.columns[2:]
    df = df[questions]
    return df


def import_3_exams_data(csvpath):
    data = pd.read_csv(csvpath)

    # for now we are not using the date

    exam_1_columns = ['Name', 'q1', 'q2', 'q3',
                      'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14',
                      'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24']
    exam_2_columns = ['Name.1', 'q1.1', 'q2.1', 'q3.1',
                      'q4.1', 'q5.1', 'q6.1', 'q7.1', 'q8.1', 'q9.1', 'q10.1', 'q11.1',
                      'q12.1', 'q13.1', 'q14.1', 'q15.1', 'q16.1', 'q17.1', 'q18.1', 'q19.1',
                      'q20.1', 'q21.1', 'q22.1', 'q23.1']
    exam_3_columns = ['Name.2', 'q1.2', 'q2.2', 'q3.2', 'q4.2', 'q5.2', 'q6.2',
                      'q7.2', 'q8.2', 'q9.2', 'q10.2', 'q11.2', 'q12.2', 'q13.2', 'q14.2',
                      'q15.2', 'q16.2', 'q17.2', 'q18.2', 'q19.2', 'q20.2', 'q21.2', 'q22.2',
                      'q23.2']

    data_1H = data[exam_1_columns]
    data_2H = data[exam_2_columns]
    data_3H = data[exam_3_columns]

    df_1, q_info_1 = clean_exam(data_1H)
    df_2, q_info_2 = clean_exam(data_2H)
    df_3, q_info_3 = clean_exam(data_3H)

    return df_1, q_info_1, df_2, q_info_2, df_3, q_info_3


def import_first_exam(csvpath):
    df_1, q_info_1, _, _, _, _ = import_3_exams_data(csvpath)
    return df_1, q_info_1


def import_second_exam(csvpath):
    _, _, df_2, q_info_2, _, _ = import_3_exams_data(csvpath)
    return df_2, q_info_2


def import_third_exam(csvpath):
    _, _, _, _, df_3, q_info_3 = import_3_exams_data(csvpath)
    return df_3, q_info_3


def read_xlsm_save_csv(file_path):
    """
    Takes the file_path (full path!) of the xlsm document, reads sheets (one sheet in this case) and returns the save location of the sheet in the .csv form
    """
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheets()[0]

    csv_save_path = os.path.join(os.getcwd(), 'data', '{}.csv'.format(sheet.name))
    with open(csv_save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(sheet.row_values(row) for row in range(sheet.nrows))

    return csv_save_path 


def load_clean_save_three_exams(raw_csv_path):
    """
    Read raw csv path (full path!) of the data for all three exams. This .csv path contains data, question dictionary and ALL students and ALL questions 
    (some students in the table didn't sit all exams)
    This function cleans the data such that only students who took all exams are retained. 
    Data is saved in the data folder under file name: 'three_exams_cleaned.csv'
    Questions information dictionary is saved in data folder under file name: 'questions_info_dict.p'

    Returns data_save_path and dictionary_save_path
    """

    three_exams_raw = pd.read_csv(raw_csv_path)

    # remove unnamed columns from the dataset
    three_exams_raw = three_exams_raw.loc[:, ~three_exams_raw.columns.str.contains('^Unnamed')]

    # remove date columns 
    three_exams_raw = three_exams_raw.loc[:, ~three_exams_raw.columns.str.contains('^date')]

    # drop name columns 
    three_exams_raw = three_exams_raw.loc[:, ~three_exams_raw.columns.str.contains('^Name')]

    # create dictionary for question, containing all questions - accross three datasets 
    questions_info_df = three_exams_raw.head(5)
    questions = list(questions_info_df.columns)
    questions_info = {}
    for question in questions:
        sup_dict = {'Actual': questions_info_df[question][0],
                    'Max': float(questions_info_df[question][1]),
                    'Topics': questions_info_df[question][2],
                    'Difficulty': questions_info_df[question][3],
                    'qtype': questions_info_df[question][4]}

        questions_info[question] = sup_dict

    # take just data, without the question dictionary 
    data = three_exams_raw[6:]

    # drop students that haven't solved all questions (accross three exams)
    data = data.dropna(axis=0)

    # check whether all NaN values are removed from the dataset 
    assert(data.isna().sum().sum() == 0)

    # cast dataframe values as float
    data = data.astype('float')

    # save data as csv file 
    data_save_path = os.path.join(os.getcwd(), 'data', 'three_exams_cleaned.csv')
    data.to_csv(data_save_path, index=False)

    # save dictionary as pkl 
    dictionary_save_path = os.path.join(os.getcwd(), 'data', 'questions_info_dict.p')
    with open(dictionary_save_path, 'wb') as fp:
        pkl.dump(questions_info, fp, protocol=pkl.HIGHEST_PROTOCOL)

    return data_save_path, dictionary_save_path


def load_data_and_dictionary(data_path, dictionary_path):
    """
    Set of calls before this function to ensure that data and dictionary are made and ready at the designated paths:

    exams_xlsm_path = os.path.join(os.getcwd(), 'data', 'three_exams.xlsm')
    csv_save_path = functions_data.read_xlsm_save_csv(exams_xlsm_path)
    data_path, dict_path = functions_data.load_clean_save_three_exams(csv_save_path)

    ---------------------------------------------------------------------------------------------------------------

    Read already cleaned data from data_path file (input full path!) and dictionary (input full path!) 
    """
    data = pd.read_csv(data_path)
    file = open(dictionary_path,'rb')
    questions_info = pkl.load(file)
    file.close()

    return data, questions_info



# select the number of students to train question parameters (and student parameters - for old students)
def convert_dataframe_to_triples(data):
    """
    input - data: dataframe of dimensions n_students x n_questions 
    returns tensor of dimensions ((n_students*n_questions), 3)
    where first n_questions rows correspond to student 1, next n_questions rows correspond to student 3, etc. 
    """
    data_tensor = torch.tensor(data.values)
    n_students, n_questions = data_tensor.shape[0], data_tensor.shape[1]

    data_tensor = data_tensor.reshape(-1)

    student_id = torch.arange(1, n_students + 1)
    student_id = student_id.repeat_interleave(n_questions)

    questions_in_order = data.columns
    question_id = []
    for q in questions_in_order:
        question_id.append(int(q[1:]))

    question_id = torch.tensor(question_id)
    
    # question_id = torch.arange(1, n_questions + 1)
    question_id = question_id.repeat(n_students)
    return torch.stack((student_id, question_id, data_tensor), dim=1)
