import pandas as pd 
import os 
import pickle as pkl 
import math 

import torch
import functions_data
import functions_ordinal
import seaborn as sns
import matplotlib.pyplot as plt 
from torch import optim
import tqdm
from functions_test_manipulations import get_test, merge_tests, get_dictionary, merge_dictionaries

# Vary this parameter in the script to select different fraction of train data and compare the accuracy on the test data 
train_fraction = 80


sigmoid = torch.nn.Sigmoid()
torch.manual_seed(100)

data_path = os.path.join(os.getcwd(), 'data', 'three_exams_cleaned.csv')
dict_path = os.path.join(os.getcwd(), 'data', 'questions_info_dict.p')
data, questions_info = functions_data.load_data_and_dictionary(data_path, dict_path)

# if the score is higher than max score for the given question, put score = max_score 
for col in data:
    m = questions_info[col]['Max']
    data[col].loc[data[col] > m] = m

# rename keys in the questions_info 
new_dictionary = {}
columns_mapper = {}
dict_keys = questions_info.keys()
for ind, key_old in enumerate(dict_keys):
    key_new = 'q' + str(ind + 1)
    new_dictionary[key_new] = questions_info[key_old]
    columns_mapper[key_old] = key_new 

# rename column in the dataframe
data = data.rename(columns=columns_mapper)


ordinal_data = data.astype('float')
ordinal_tensor = torch.tensor(ordinal_data.values)

n_students, n_questions = ordinal_tensor.shape[0], ordinal_tensor.shape[1]
max_score = int(ordinal_tensor.max().numpy())

train_data, validation_data, test_data = functions_ordinal.custom_separate_train_validation_test_data(ordinal_tensor, random_seed = 100, train_fraction=0.8)


train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, new_dictionary)
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, new_dictionary)
validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, new_dictionary)


# initialization of parameters 
bs = torch.randn(n_students, requires_grad=True)
bq0 = torch.distributions.uniform.Uniform(0,1).sample([n_questions,])
rho = torch.normal(0, 0.1, size = (n_questions, max_score - 1), requires_grad=True) # rho1, rho2, rho3 

bq0.requires_grad = True  


nlls_train = []
nlls_validation = []
nlls_test = []

iterations = 3000
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

# save trained parameters 
torch.save(bs, os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'bs_test{train_fraction}.pth'))
torch.save(bq0, os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'bq0_test{train_fraction}.pth'))
torch.save(rho, os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'rho_test{train_fraction}.pth'))

# save array of losses 
torch.save(nlls_train, os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'nlls_train_test{train_fraction}.npy'))
torch.save(nlls_validation, os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'nlls_validation_test{train_fraction}.npy'))
torch.save(nlls_test, os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'nlls_test_test{train_fraction}.npy'))

print('Done with executing the script')



