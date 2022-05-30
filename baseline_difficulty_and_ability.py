import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.random import default_rng
import os

from torch import threshold 

from functions_data import binarise_using_average, binarise_using_max, import_data, shuffle_data

def question_difficulty_model(non_binarised_data, questions_info, binarisation_method, shuffle, thresholding = False):

    if shuffle:
        non_binarised_data = shuffle_data(non_binarised_data)  # shuffle columns of the dataset

    if binarisation_method == 'average':
        data = binarise_using_average(non_binarised_data)
    if binarisation_method == 'max':
        data = binarise_using_max(non_binarised_data, questions_info)

    students_separation_point = 22511
    questions_separation_point = 12
    question_labels = data.columns

    test = data[question_labels[questions_separation_point:]][students_separation_point:]
    train = data[question_labels[questions_separation_point:]][:students_separation_point]
    test.reset_index(inplace=True, drop=True)
    train.reset_index(inplace=True, drop=True)
    
    p = train.sum(axis=0) / train.shape[0]

    if thresholding:
        # thresholding 
        correct = 0
        for col in test:
            pred = np.repeat(int(p[col] > 0.5), test.shape[0])
            correct += np.count_nonzero(pred == test[col])
        correctness = correct / test.size * 100
    else:
        # no thresholding, generates predictions from the Bernouli distribution given the question parameter 
        correctness = []
        for seed_number in range(100):
            rng = default_rng(seed=seed_number)
            correct = 0
            for col in test:
                pred = rng.binomial(1, p[col], test.shape[0])
                correct += np.count_nonzero(pred == test[col])
            correctness.append(correct / test.size * 100)
    
    
    acc, std = np.mean(correctness), np.std(correctness)

    return acc, std


def student_ability_model(non_binarised_data, questions_info, binarisation_method, shuffle, thresholding = False):

    if shuffle:
        non_binarised_data = shuffle_data(non_binarised_data, 1000)  # shuffle columns of the dataset

    if binarisation_method == 'average':
        data = binarise_using_average(non_binarised_data)
    if binarisation_method == 'max':
        data = binarise_using_max(non_binarised_data, questions_info)

    students_separation_point = 22511
    questions_separation_point = 12
    question_labels = data.columns

    test = data[question_labels[questions_separation_point:]][
           students_separation_point:]  # test set will be 13 questions
    train = data[question_labels[:questions_separation_point]][students_separation_point:]
    test.reset_index(inplace=True, drop=True)
    train.reset_index(inplace=True, drop=True)

    p = train.sum(axis=1) / train.shape[1]

    if thresholding:
        correct = 0
        for index, student in test.iterrows():
            pred = np.repeat(int(p[index]>0.5), test.shape[1])
            correct += np.count_nonzero(pred == student)
        correctness = correct / test.size * 100
    else: 
        # no thresholding 
        correctness = []
        for seed_number in range(10):
            correct = 0
            rng = default_rng(seed=seed_number)
            for index, student in test.iterrows():
                pred = rng.binomial(1, p[index], test.shape[1])
                correct += np.count_nonzero(pred == student)
            correctness.append(correct / test.size * 100)    

    acc, std = np.mean(correctness), np.std(correctness)

    return acc, std


def corr_matrix_questions(data, save_location):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.savefig(save_location)
    plt.show()


def zeros_ones_data(data, questions_info):
    binarised_using_avg = binarise_using_average(data)
    binarised_using_max = binarise_using_max(data, questions_info)

    bin_avg_frac = binarised_using_avg.values.sum() / binarised_using_avg.size * 100
    bin_max_frac = binarised_using_max.values.sum() / binarised_using_max.size * 100

    print('Fraction of ones in the dataset binarised using average of the question score is ',
          "{0:0.2f}".format(bin_avg_frac), "%")
    print('Fraction of ones in the dataset binarised using max possible question score is ',
          "{0:0.2f}".format(bin_max_frac), "%")


def run_model(data, questions_info, model, binarisation_method, shuffle, thresholding):

    if model == 'question_difficulty':
        acc, std = question_difficulty_model(data, questions_info, binarisation_method, shuffle, thresholding)
        if shuffle:
            print('Question difficulty model with shuffled columns accuracy: ', acc, ' standard deviation: ', std,
                  'binarisation method ', binarisation_method)
        else:
            print('Question difficulty model without shuffled columns accuracy: ', acc, ' standard deviation: ', std,
                  'binarisation method ', binarisation_method)

    if model == 'student_ability':
        acc, std = student_ability_model(data, questions_info, binarisation_method=binarisation_method, shuffle=shuffle, thresholding = thresholding)
        if shuffle:
            print('Student ability model with shuffled columns accuracy: ', acc, ' standard deviation: ', std,
                  'binarisation method ', binarisation_method)
        else:
            print('Student ability model without shuffled columns accuracy: ', acc, ' standard deviation: ', std,
                  'binarisation method ', binarisation_method)

data_file = os.path.join(os.getcwd(), 'data', '9to1_2017_GCSE_1H.csv')
df, q_info = import_data(data_file)

model_ = 'student_ability'
shuffle_ = True   
binarisation_method_ = 'max'
thresholding_ = False  
run_model(df, q_info, model_, binarisation_method_, shuffle_, thresholding_)

# accs = []
# sep_points = np.arange(100, 22511, 100)
# for sep_point in sep_points:
#     acc = run_model(df, q_info, model_, binarisation_method_, shuffle_, sep_point)
#     accs.append(acc)
#     print(sep_point)
# plt.plot(sep_points, accs)
# plt.xlabel('Number of students in the training set')
# plt.ylabel('Accuracy of the question-difficulty model')
# plt.show()

