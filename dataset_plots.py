from cProfile import label
from turtle import left
import torch 
import os 
from functions_data import load_data_and_dictionary, convert_dataframe_to_triples
import functions_ordinal
import numpy as np 
from scipy.stats import entropy
import pandas as pd 
from torch import optim
import matplotlib.pyplot as plt 
from functions_data import binarise_using_max, import_data
from collections import Counter


data_file = os.path.join(os.getcwd(), 'data', '9to1_2017_GCSE_1H.csv')
data, new_dictionary = import_data(data_file)

## code for three exams 
# data_path = os.path.join(os.getcwd(), 'data', 'three_exams_cleaned.csv')
# dict_path = os.path.join(os.getcwd(), 'data', 'questions_info_dict.p')
# data, questions_info = load_data_and_dictionary(data_path, dict_path)

# n_students, n_questions = data.shape[0], data.shape[1]

# # if the score is higher than max score for the given question, put score = max_score 
# for col in data:
#     m = questions_info[col]['Max']
#     data[col].loc[data[col] > m] = m

# # rename keys in the questions_info 
# new_dictionary, columns_mapper = {}, {}
# dict_keys = questions_info.keys()
# for ind, key_old in enumerate(dict_keys):
#     key_new = 'q' + str(ind + 1)
#     new_dictionary[key_new] = questions_info[key_old]
#     columns_mapper[key_old] = key_new 

# # rename column in the dataframe
# data = data.rename(columns=columns_mapper)
max_score = int(data.values.max())

data_values = data.values.flatten()


binary_data = binarise_using_max(data, new_dictionary)
binary_data_values = binary_data.values.flatten()

labels_ordinal, counts_ordinal = [], []
counter = Counter(data_values)
for i in range(max_score + 1):
    labels_ordinal.append(str(i))
    counts_ordinal.append(counter[i])

labels_binary, counts_binary = [], []
counter = Counter(binary_data_values)
for i in range(2):
    labels_binary.append(str(i))
    counts_binary.append(counter[i])


fig, ax = plt.subplots(1,2)
fig.set_figwidth(12), fig.set_figheight(5)
ax[0].bar(labels_ordinal, counts_ordinal, color = 'lightgreen')
ax[1].bar(labels_binary, counts_binary, color = 'lightblue')

fig.suptitle('Count of scores in data sets', fontsize = 14)
ax[1].set_title('Binary data set', fontsize = 14)
ax[0].set_title('Ordinal data set', fontsize = 14)
ax[0].set_ylabel('count', fontsize = 14)
ax[1].set_xlabel('binary score', fontsize = 14)
ax[0].set_xlabel('ordinal score', fontsize = 14)
plt.show()

# d = np.diff(np.unique(data_values)).min()
# left_of_first_bin = data_values.min() - float(d)/2
# right_of_last_bin = data_values.max() + float(d)/2
# plt.hist(data_values, np.arange(left_of_first_bin, right_of_last_bin + d, d), color = 'lightgreen')
# plt.title('Count of ordinal scores', fontsize = 14)
# plt.xlabel('score on the question', fontsize = 14)
# plt.ylabel('count', fontsize = 14)
# plt.ticklabel_format(axis = 'y', style='sci', scilimits=(0,0))
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.show()





# zeros = binary_data_values.shape[0] - binary_data_values.sum()
# ones = binary_data_values.sum()
# x = [zeros, ones]

# plt.bar(['0', '1'], height=x,color=['lightgreen','lightcoral'])
# plt.title('Count of binary scores', fontsize = 14)
# plt.ticklabel_format(axis = 'y', style='sci', scilimits=(0,0))
# plt.yticks(fontsize=14)
# plt.ylabel('count', fontsize=14)
# plt.xticks(fontsize=14)
# plt.show()

