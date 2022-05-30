import torch 
import os 
from functions_data import load_data_and_dictionary, convert_dataframe_to_triples
import functions_ordinal
import numpy as np 
from scipy.stats import entropy
import pandas as pd 
from torch import optim
import matplotlib.pyplot as plt 

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

def reveal_data(revealed_data, hidden_data, bs, bq0, rho, max_score, revealed_max_scores, hidden_max_scores):
    # data from which to choose 1 most uncertain question per student 
    prob_matrix = functions_ordinal.generate_prob_matrix(hidden_data, bs, bq0, rho, max_score, hidden_max_scores).detach()
    entropies = entropy(prob_matrix.T)

    # convert to dataframe and add fourth column for the entropy 
    df = pd.DataFrame(hidden_data.numpy(), columns=['studentID', 'questionID', 'score'])
    df['entropy'] = entropies

    # get the most uncertain question for each student to be revealed (uncertainty quantified by highest entropy for the distribution for a particular student)
    idx = df.groupby(['studentID'])['entropy'].transform(max) == df['entropy']

    # most uncertain question for each student, indices, all revealed indices so far
    reveal_indices = list(df[idx].index)

    # get hidden indices 
    set_reveal_indices = set(reveal_indices)
    set_all_indices = set(list(np.arange(0, hidden_data.shape[0], 1)))
    set_hide_indices = set_all_indices - set_reveal_indices
    hide_indices = list(set_hide_indices)

    revealed_data = torch.cat((revealed_data, hidden_data[reveal_indices]), 0)
    hidden_data = hidden_data[hide_indices]
    revealed_max_scores = torch.cat((revealed_max_scores, hidden_max_scores[reveal_indices]), 0)
    hidden_max_scores = hidden_max_scores[hide_indices]

    return revealed_data, revealed_max_scores, hidden_data, hidden_max_scores

def retrain_parameters(data, data_max_scores, bs_pool, bq0_pool, rho_pool, max_score, loss_, it):
    # if it < 10:
    #     iterations = 1000
    # else: 
    #     iterations = 100

    iterations = 500 

    params = [bs_pool, bq0_pool, rho_pool]

    opt = optim.SGD(params, lr = 0.0001)
    n = data.shape[0]

    for iter in range(iterations):
        loss = functions_ordinal.nll(data, params, max_score, data_max_scores)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_.append(loss/n)
    
    return bs_pool, bq0_pool, rho_pool, loss_



# Perform entropy pooling learning 

loss_entropies = []
accuracies_entropy = [initial_accuracy]

hidden_data = learning_pool.clone()
revealed_data = torch.tensor([])
hidden_max_scores = learning_pool_max_scores.clone()
revealed_max_scores = torch.tensor([])


for it in range(int(active_learning_iterations/2)):

    revealed_data, revealed_max_scores, hidden_data, hidden_max_scores = reveal_data(revealed_data, hidden_data, bs_pool, bq0_pool, rho_pool, max_score, revealed_max_scores, hidden_max_scores)
    revealed_data, revealed_max_scores, hidden_data, hidden_max_scores = reveal_data(revealed_data, hidden_data, bs_pool, bq0_pool, rho_pool, max_score, revealed_max_scores, hidden_max_scores)

    # revealed_data, revealed_max_scores, hidden_data, hidden_max_scores = reveal_data(revealed_data, hidden_data, bs_pool, bq0_pool, rho_pool, max_score, revealed_max_scores, hidden_max_scores)

    bs_pool, bq0_pool, rho_pool, loss_entropies = retrain_parameters(revealed_data, revealed_max_scores, bs_pool, bq0_pool, rho_pool, max_score, loss_entropies, it)

    prob_matrix = functions_ordinal.generate_prob_matrix(testing_pool, bs_pool, bq0_pool, rho_pool, max_score, testing_pool_max_scores)
    predicted_scores = prob_matrix.argmax(axis = 1)
    true_scores = testing_pool[:, 2]
    accuracies_entropy.append(functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy())
    
    if it % 10 == 0:
        print(it)


prob_matrix = functions_ordinal.generate_prob_matrix(testing_pool, bs_pool, bq0_pool, rho_pool, max_score, testing_pool_max_scores)
predicted_scores = prob_matrix.argmax(axis = 1)
true_scores = testing_pool[:, 2]

accuracy_post_training = functions_ordinal.full_accuracy(true_scores, predicted_scores)
print(f'Accuracy on the testing pool after the training is {accuracy_post_training}')

plt.plot(accuracies_entropy)

plt.show()