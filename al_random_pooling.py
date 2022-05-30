import torch 
import os 
from functions_data import load_data_and_dictionary, convert_dataframe_to_triples
import functions_ordinal
import numpy as np 
from scipy.stats import entropy
import pandas as pd 
from torch import optim
import matplotlib.pyplot as plt 
from functions_active_learning import retrain_parameters, random_reveal_questions_per_student

data_path = os.path.join(os.getcwd(), 'data', 'three_exams_cleaned.csv')
dict_path = os.path.join(os.getcwd(), 'data', 'questions_info_dict.p')
data, questions_info = load_data_and_dictionary(data_path, dict_path)

n_students, n_questions = data.shape[0], data.shape[1]

# if the score is higher than max score for the given question, put score = max_score 
for col in data:
    m = questions_info[col]['Max']
    data[col].loc[data[col] > m] = m

# rename keys in the questions_info 
new_dictionary, columns_mapper = {}, {}
dict_keys = questions_info.keys()
for ind, key_old in enumerate(dict_keys):
    key_new = 'q' + str(ind + 1)
    new_dictionary[key_new] = questions_info[key_old]
    columns_mapper[key_old] = key_new 

# rename column in the dataframe
data = data.rename(columns=columns_mapper)
max_score = int(data.values.max())


# separate training and pooling students 
training_students = 13563
pooling_students = data.shape[0] - training_students
training_dataframe = data[:training_students]
pooling_dataframe = data[training_students:]


# in the pooling dataframe separate learning and testing data frame
def shuffle_data(raw_data, shuffle_seed=1000):
    df = raw_data.sample(frac=1, axis=1, random_state=np.random.RandomState(shuffle_seed))
    return df
shuffled_pooling_dataframe = shuffle_data(pooling_dataframe)
questions = list(shuffled_pooling_dataframe.columns)
learning_questions = questions[:50]
testing_questions = questions[50:]
learning_pool_df = shuffled_pooling_dataframe[learning_questions]
testing_pool_df = shuffled_pooling_dataframe[testing_questions]
learning_pool = convert_dataframe_to_triples(learning_pool_df)
testing_pool = convert_dataframe_to_triples(testing_pool_df)


#load pretrained parameters 
bs_pretrained = torch.load(os.path.join(os.getcwd(), 'al_params', 'bs_pretrained.pth'))
bq0_pretrained = torch.load(os.path.join(os.getcwd(), 'al_params', 'bq0_pretrained.pth'))
rho_pretrained = torch.load(os.path.join(os.getcwd(), 'al_params', 'rho_pretrained.pth'))


# ACTIVE LEARNING 
s_lp = learning_pool_df.shape[0] # number of students in the learning pool
q_lp = learning_pool_df.shape[1] # number of questions in the learning pool 

torch.manual_seed(1000)
bs_pool = torch.randn(s_lp, requires_grad=True)
bq0_pool = bq0_pretrained
rho_pool = rho_pretrained

active_learning_iterations = q_lp # in every iteration a single question to each student is revealed 

# predictions on the testing pool before the training 
testing_pool_max_scores = functions_ordinal.generate_max_scores_tensor(testing_pool, new_dictionary)
learning_pool_max_scores = functions_ordinal.generate_max_scores_tensor(learning_pool, new_dictionary)

prob_matrix = functions_ordinal.generate_prob_matrix(testing_pool, bs_pool, bq0_pool, rho_pool, max_score, testing_pool_max_scores)
predicted_scores = prob_matrix.argmax(axis = 1)
true_scores = testing_pool[:, 2]

initial_accuracy = functions_ordinal.full_accuracy(true_scores, predicted_scores)
print(f'Accuracy on the testing pool before the training is {initial_accuracy}')


# ACTIVE LEARNING for RANDOM pooling 

# pre-requisites for AL 

s_lp = learning_pool_df.shape[0] # number of students in the learning pool
q_lp = learning_pool_df.shape[1] # number of questions in the learning pool 

# separate labelled and unlabelled data; for beginning labelled data is none data and unlabelled data is all data in the learning pool 
labelled_pool = torch.tensor([])
unlabelled_pool = learning_pool.clone()

labelled_max_scores = torch.tensor([])
unlabelled_max_scores = learning_pool_max_scores.clone()

# Random selection of a question per student revealed 
loss_random = []
accs_random = [initial_accuracy]

random_seed = 1004
questions_per_student = 1 
iterations = int(q_lp / questions_per_student) # number of retraining iterations based on how many questions are revealed to each student in every iteration
params = [bs_pool, bq0_pool, rho_pool]
labelled = [labelled_pool, labelled_max_scores]
unlabelled = [unlabelled_pool, unlabelled_max_scores]

print(f'I will be training now for {iterations}')
bs_random_1= []
bs_random_2 = []
bs_random_3 = []
bs_random_4 = []

for it in range(iterations):
    labelled, unlabelled = random_reveal_questions_per_student(labelled, unlabelled, questions_per_student, random_seed)
    labelled_data, labelled_data_max_scores = labelled[0], labelled[1]

    params, loss_random = retrain_parameters(labelled_data, labelled_data_max_scores, params, max_score, loss_random, it)

    prob_matrix = functions_ordinal.generate_prob_matrix(testing_pool, bs_pool, bq0_pool, rho_pool, max_score, testing_pool_max_scores)
    predicted_scores, true_scores = prob_matrix.argmax(axis = 1), testing_pool[:, 2]
    accs_random.append(functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy())
    
    bs_random_1.append(bs_pool[1].clone().detach().numpy())
    bs_random_2.append(bs_pool[2].clone().detach().numpy())
    bs_random_3.append(bs_pool[3].clone().detach().numpy())
    bs_random_4.append(bs_pool[4].clone().detach().numpy())

    if it % 10 == 0:
        print(it)

torch.save(accs_random, os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_accs_long_training.pth'))
torch.save(loss_random, os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_loss_long_training.pth'))
torch.save(bs_random_1, os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs1.pth'))
torch.save(bs_random_2, os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs2.pth'))
torch.save(bs_random_3, os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs3.pth'))
torch.save(bs_random_4, os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs4.pth'))

print('Done with executing the script for random pooling')