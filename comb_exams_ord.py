from heapq import merge
from time import time
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
'9to1_2017_GCSE_3H', '9to1_Edexcel_GCSE_Nov2018_1H', '9to1_Edexcel_GCSE_Nov2018_2H', 
'9to1_Edexcel_GCSE_Nov2018_3H', '9to1_Edexcel_Nov2017_GCSE_2H', '9to1_Edexcel_Nov2017_GCSE_3H']


df_merged, dict_merged, _ = get_merged_test_and_dict(test_names)
ordinal_data = df_merged.astype('float')
ordinal_tensor = torch.tensor(ordinal_data.values)

n_students, n_questions = ordinal_tensor.shape[0], ordinal_tensor.shape[1]
max_score = int(ordinal_tensor.max().numpy())

train_data, validation_data, test_data = functions_ordinal.separate_train_validation_test_data(ordinal_tensor)

train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, dict_merged)
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, dict_merged)
validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, dict_merged)


# initialization of parameters 
random_seed = 1008
torch.random.manual_seed(random_seed)
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

bs_1 = []
bs_2 = []
bs_3 = []


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

    bs_1.append(params[0][10].clone().detach().numpy())
    bs_2.append(params[0][16].clone().detach().numpy())
    bs_3.append(params[0][25].clone().detach().numpy())

    if iter % 100 == 0:
        print(iter)

    

plt.plot(bs_1)
plt.plot(bs_2)
plt.plot(bs_3)
plt.show()

torch.save(bs_1, os.path.join(os.getcwd(), 'params', 'bs_conv1_init_seed_1009.pth'))
torch.save(bs_2, os.path.join(os.getcwd(), 'params', 'bs_conv2_init_seed_1009.pth'))
torch.save(bs_3, os.path.join(os.getcwd(), 'params', 'bs_conv3_init_seed_1009.pth'))

# save trained parameters 
# torch.save(bs, os.path.join(os.getcwd(), 'params', 'bs_trained_4000iter_8T.pth'))
# torch.save(bq0, os.path.join(os.getcwd(), 'params', 'bq0_trained_4000iter_8T.pth'))
# torch.save(rho, os.path.join(os.getcwd(), 'params', 'rho_trained_4000iter_8T.pth'))

# # save array of losses 
# torch.save(nlls_train, os.path.join(os.getcwd(), 'params', 'nlls_train_8T.npy'))
# torch.save(nlls_validation, os.path.join(os.getcwd(), 'params', 'nlls_validation_8T.npy'))
# torch.save(nlls_test, os.path.join(os.getcwd(), 'params', 'nlls_test_8T.npy'))

print('Done with executing the script')





