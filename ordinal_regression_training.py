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


sigmoid = torch.nn.Sigmoid()
torch.manual_seed(100)


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

test_name_1H = '9to1_2017_GCSE_1H'
test_name_2H = '9to1_2017_GCSE_2H'

test_names = [test_name_1H, test_name_2H]

test_m = merge_tests(data, test_names, 'AnonymousStudentID')
dict_m = merge_dictionaries(meta_dict, test_names)

student_IDs_original = test_m['AnonymousStudentID']

# drop column in test_m that has just studentIDs 
if 'AnonymousStudentID' in test_m.columns:
    test_m.drop(columns=['AnonymousStudentID'], inplace=True)
    
for col in test_m:
    max_score = dict_m[col]['Max']
    test_m[col].loc[test_m[col] > max_score] = max_score


ordinal_data = test_m.astype('float')
ordinal_tensor = torch.tensor(ordinal_data.values)

n_students, n_questions = ordinal_tensor.shape[0], ordinal_tensor.shape[1]
max_score = int(ordinal_tensor.max().numpy())

train_data, validation_data, test_data = functions_ordinal.separate_train_validation_test_data(ordinal_tensor, random_seed = 100)

train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, dict_m)
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, dict_m)
validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, dict_m)


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


# save trained parameters 
torch.save(bs, os.path.join(os.getcwd(), 'params', 'bs_trained_4000iter.pth'))
torch.save(bq0, os.path.join(os.getcwd(), 'params', 'bq0_trained_4000iter.pth'))
torch.save(rho, os.path.join(os.getcwd(), 'params', 'rho_trained_4000iter.pth'))

# save array of losses 
torch.save(nlls_train, os.path.join(os.getcwd(), 'params', 'nlls_train.npy'))
torch.save(nlls_validation, os.path.join(os.getcwd(), 'params', 'nlls_validation.npy'))
torch.save(nlls_test, os.path.join(os.getcwd(), 'params', 'nlls_test.npy'))

print('Done with executing the script')



