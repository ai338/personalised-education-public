from heapq import merge
import matplotlib.pyplot as plt 
import torch 
import os 
import numpy as np 
from functions_test_manipulations import merge_dictionaries, merge_tests, get_merged_test_and_dict
import functions_ordinal
import pickle as pkl 
import pandas as pd 
from torch import optim


test_names = ['9to1_2017_GCSE_1H', '9to1_2017_GCSE_2H', 
'9to1_2017_GCSE_3H']

df_merged, dict_merged, _ = get_merged_test_and_dict(test_names)
ordinal_data = df_merged.astype('float')
ordinal_tensor = torch.tensor(ordinal_data.values)

n_students, n_questions = ordinal_tensor.shape[0], ordinal_tensor.shape[1]
max_score = int(ordinal_tensor.max().numpy())

random_seeds = np.arange(1000, 1020, 1)
print(random_seeds)

accuracies = []
for random_seed in random_seeds:

    train_data, validation_data, test_data = functions_ordinal.separate_train_validation_test_data(ordinal_tensor, random_seed)

    train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, dict_merged)
    test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, dict_merged)
    validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, dict_merged)


    # initialization of parameters 
    bs = torch.randn(n_students, requires_grad=True)
    bq0 = torch.distributions.uniform.Uniform(0,1).sample([n_questions,])
    rho = torch.normal(0, 0.1, size = (n_questions, max_score - 1), requires_grad=True) # rho1, rho2, rho3 

    bq0.requires_grad = True  


    nlls_train = []
    nlls_validation = []
    nlls_test = []

    iterations = 4000
    params = [bs, bq0, rho]

    opt = optim.SGD(params, lr = 0.0001)

    for iter in range(iterations):
        loss = functions_ordinal.nll(train_data, params, max_score, train_data_max_scores)    
        opt.zero_grad()
        loss.backward()
        opt.step()
        nlls_train.append(loss.detach().numpy())

        loss_validation = functions_ordinal.nll(validation_data, params, max_score, validation_data_max_scores)
        loss_test = functions_ordinal.nll(test_data, params, max_score, test_data_max_scores)

        nlls_validation.append(loss_validation.detach().numpy())
        nlls_test.append(loss_test.detach().numpy())

        if iter % 100 == 0:
            print(iter)
    
    prob_matrix = functions_ordinal.generate_prob_matrix(test_data, bs, bq0, rho, max_score, test_data_max_scores)

    predicted_scores_thresholding = prob_matrix.argmax(axis = 1)
    predicted_scores = predicted_scores_thresholding


    true_scores = test_data[:, 2]
    accuracy = functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy()
    accuracies.append(accuracy)

    print('Accuracy for random seed {} is {}'.format(random_seed, accuracy))

torch.save(accuracies, os.path.join(os.getcwd(), 'params', 'accuracies_2017_3T_20seeds.npy'))
print('Done with executing the script')





