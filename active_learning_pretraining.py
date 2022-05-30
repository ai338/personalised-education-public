import torch
import functions_data
import os 
import functions_ordinal
import functions_plotting
import seaborn as sns
import matplotlib.pyplot as plt 
from torch import optim
import tqdm
from scipy.stats import entropy
import pandas as pd 
import numpy as np 

# load data and information dictionary 
data_path = os.path.join(os.getcwd(), 'data', 'three_exams_cleaned.csv')
dict_path = os.path.join(os.getcwd(), 'data', 'questions_info_dict.p')
data, questions_info = functions_data.load_data_and_dictionary(data_path, dict_path)

n_students, n_questions = data.shape[0], data.shape[1]

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

max_score = int(data.values.max())

training_students = 13563
pooling_students = data.shape[0] - training_students

training_dataframe = data[:training_students]
pooling_dataframe = data[training_students:]

torch.manual_seed(100)

n_students_training = training_students
n_questions_training = n_questions

# initialization of parameters 
bs = torch.randn(n_students_training, requires_grad=True)
bq0 = torch.randn([n_questions_training,], requires_grad=True)
rho = torch.normal(0, 0.1, size = (n_questions_training, max_score - 1), requires_grad=True) # rho1, rho2, rho3 

nlls_train, nlls_validation, nlls_test = [], [], []

torch.random.manual_seed(100)

# Shuffle training data and separate train test and validation data within the training data for active learning 
data_training_triplets = functions_data.convert_dataframe_to_triples(training_dataframe)
shuffled_train_pool_data = data_training_triplets[torch.randperm(data_training_triplets.size()[0])]
n_datapoints = shuffled_train_pool_data.shape[0]

# 80/10/10 train/test/validation split
train_data_tp = shuffled_train_pool_data[:int(0.8*n_datapoints)]
test_data_tp = shuffled_train_pool_data[int(0.8*n_datapoints):int(0.9*n_datapoints)]
validation_data_tp = shuffled_train_pool_data[int(0.9*n_datapoints):]


# max scores for train, test and validation in the training pool 
tp_train_max = functions_ordinal.generate_max_scores_tensor(train_data_tp, new_dictionary)
tp_valid_max = functions_ordinal.generate_max_scores_tensor(validation_data_tp, new_dictionary)
tp_test_max = functions_ordinal.generate_max_scores_tensor(test_data_tp, new_dictionary)

iterations = 4000
params = [bs, bq0, rho]
train_loss, test_loss, valid_loss = [], [], []

opt = optim.SGD(params, lr = 0.0001)
for iter in range(iterations):
    loss = functions_ordinal.nll(train_data_tp, params, max_score, tp_train_max)    
    opt.zero_grad()
    loss.backward()
    opt.step()
    train_loss.append(loss.detach().numpy())

    valid_loss.append(functions_ordinal.nll(validation_data_tp, params, max_score, tp_valid_max).detach().numpy())
    test_loss.append(functions_ordinal.nll(test_data_tp, params, max_score, tp_test_max).detach().numpy())


    if iter % 100 == 0:
        print(f'iteration {iter} | NLL {loss.detach().numpy()}')

# save parameters for active learning pretraining
# save trained parameters 
torch.save(bs, os.path.join(os.getcwd(), 'al_params', 'bs_pretrained.pth'))
torch.save(bq0, os.path.join(os.getcwd(), 'al_params', 'bq0_pretrained.pth'))
torch.save(rho, os.path.join(os.getcwd(), 'al_params', 'rho_pretrained.pth'))

# save array of losses 
torch.save(train_loss, os.path.join(os.getcwd(), 'al_params', 'nlls_train_pretrained.npy'))
torch.save(valid_loss, os.path.join(os.getcwd(), 'al_params', 'nlls_valid_pretrained.npy'))
torch.save(test_loss, os.path.join(os.getcwd(), 'al_params', 'nlls_test_pretrained.npy'))

print('Done with executing the script')