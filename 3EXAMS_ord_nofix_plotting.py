import matplotlib.pyplot as plt 
import torch 
import os 
import numpy as np 
from matplotlib.ticker import FormatStrFormatter
from functions_test_manipulations import get_test, get_dictionary, merge_dictionaries, merge_tests
import functions_ordinal
import pickle as pkl 
import pandas as pd 
import functions_data


# load trained parameters 
bs_trained = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_bs_nofixing_prob.pth'))
bq0_trained = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_bq0_nofixing_prob.pth'))
rho_trained = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_rho_nofixing_prob.pth'))

# load loss values for training, testing and validation data 
nlls_train = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_nlls_train_nofixing_prob.npy'))
nlls_validation = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_nlls_validation_nofixing_prob.npy'))
nlls_test = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_nlls_test_nofixing_prob.npy'))


nlls_train = [x/0.8 for x in nlls_train]
nlls_test = [x/0.1 for x in nlls_test]
nlls_validation = [x/0.1 for x in nlls_validation]

plt.plot(nlls_train, label = 'train loss')
plt.plot(nlls_test, label = 'test loss')
plt.plot(nlls_validation, label  = 'valid. loss')
plt.xlabel('iteration', fontsize = 13)
plt.ylabel('loss', fontsize = 13)

plt.legend()
plt.show()

# LOAD DATA TO MAKE PREDICTIONS 
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


# convert pandas dataframe to tensor 
data_tensor = torch.tensor(data.values)

# separate train, test and validation data 
train_data, validation_data, test_data = functions_ordinal.separate_train_validation_test_data(data_tensor)

n_students, n_questions = data_tensor.shape[0], data_tensor.shape[1]
max_score = int(data_tensor.max().numpy())

test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, new_dictionary)
train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, new_dictionary)
validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, new_dictionary)


prob_matrix = functions_ordinal.generate_prob_matrix(test_data, bs_trained, bq0_trained, rho_trained, max_score, test_data_max_scores)

predicted_scores_thresholding = prob_matrix.argmax(axis = 1)
predicted_scores = predicted_scores_thresholding


# predicted_scores = functions_ordinal.predict_scores(test_data, bs_trained, bq0_trained, rho_trained, max_score, test_data_max_scores)
true_scores = test_data[:, 2]

dim = 15 
pred_mat = predicted_scores[10000:10000+dim*dim].reshape((dim,dim))
true_mat = true_scores[10000:10000+dim*dim].reshape((dim,dim))

_min, _max = 0, max_score
fig, ax = plt.subplots(1,3)
fig.set_figheight(15)
fig.set_figwidth(15)

ax[0].matshow(true_mat.detach().numpy(),cmap=plt.cm.YlGn,vmin = _min, vmax = _max)
ax[0].set_title('True scores in the test set', fontsize = 14)


ax[1].matshow(pred_mat.detach().numpy(),cmap=plt.cm.YlGn,vmin = _min, vmax = _max)
ax[1].set_title('Predicted scores in the test set', fontsize = 14)



diff_mat = torch.abs(pred_mat - true_mat)
ax[2].matshow(diff_mat.detach().numpy(),cmap=plt.cm.YlGn,vmin = _min, vmax = _max)
ax[2].set_title('Absolute error matrix', fontsize=14)

plt.show()

print('Accuracy on the test data after training is', functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy())