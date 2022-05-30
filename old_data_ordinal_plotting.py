from distutils.log import error
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
train_fraction = 70
test_fraction = 10
validation_fraction = 10
bs_trained = torch.load(os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'bs_test{train_fraction}.pth'))
bq0_trained = torch.load(os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'bq0_test{train_fraction}.pth'))
rho_trained = torch.load(os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'rho_test{train_fraction}.pth'))

# load loss values for training, testing and validation data 
nlls_train = torch.load(os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'nlls_train_test{train_fraction}.npy'))
nlls_validation = torch.load(os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'nlls_validation_test{train_fraction}.npy'))
nlls_test = torch.load(os.path.join(os.getcwd(), 'ordinal_params_3EXAMS', f'nlls_test_test{train_fraction}.npy'))




nlls_train = [x/train_fraction for x in nlls_train]
nlls_test = [x/test_fraction for x in nlls_test]
nlls_validation = [x/validation_fraction for x in nlls_validation]

plt.plot(nlls_train, label = 'train loss')
plt.plot(nlls_test, label = 'test loss')
plt.plot(nlls_validation, label  = 'valid. loss')
plt.xlabel('iteration', fontsize = 13)
plt.ylabel('loss', fontsize = 13)
plt.legend()
plt.show()


# import train/test/validation data 
# load meta data dictionary 
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

train_data, validation_data, test_data = functions_ordinal.custom_separate_train_validation_test_data(ordinal_tensor, random_seed = 100, train_fraction=0.7)


train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, new_dictionary)
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, new_dictionary)
validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, new_dictionary)

# Get accuracies and predictions on the test data 
prob_matrix = functions_ordinal.generate_prob_matrix(test_data, bs_trained, bq0_trained, rho_trained, max_score, test_data_max_scores)

predicted_scores_thresholding = prob_matrix.argmax(axis = 1)
predicted_scores = predicted_scores_thresholding


true_scores = test_data[:, 2]

dim = 15 
pred_mat = predicted_scores[10000:10000+dim*dim].reshape((dim,dim))
true_mat = true_scores[10000:10000+dim*dim].reshape((dim,dim))

_min, _max = 0, max_score
fig, ax = plt.subplots(1,3)
fig.set_figheight(5)
fig.set_figwidth(10)

ax[0].matshow(true_mat.detach().numpy(),cmap=plt.cm.YlGn,vmin = _min, vmax = _max)
ax[0].set_title('True scores in the test set')


ax[1].matshow(pred_mat.detach().numpy(),cmap=plt.cm.YlGn,vmin = _min, vmax = _max)
ax[1].set_title('Predicted scores in the test set')



diff_mat = torch.abs(pred_mat - true_mat)
ax[2].matshow(diff_mat.detach().numpy(),cmap=plt.cm.YlGn,vmin = _min, vmax = _max)
ax[2].set_title('Absolute error matrix')
plt.show()

print('Accuracy on the test data after training is', functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy())
print('Up to a point accuracy on the test data after training is', functions_ordinal.one_point_off_accuracy(true_scores, predicted_scores))

# data.shape = (12305, 47)
# Accuracy on the test data is 0.659 after training parameters 

# train_fraction = 0.8 -> Accuracy on the test data after training is 0.658989
# train_fraction = 0.7 -> Accuracy on the test data after training is 0.65859824
# train_fraction = 0.6 -> Accuracy on the test data after training is 0.65798634
# train_fraction = 0.5 -> Accuracy on the test data after training is 0.6562758
# train_fraction = 0.4 -> Accuracy on the test data after training is 0.6537764
# train_fraction = 0.3 -> Accuracy on the test data after training is 0.64840895
# train_fraction = 0.2 -> Accuracy on the test data after training is 0.6371284
# train_fraction = 0.1 -> Accuracy on the test data after training is 0.6097013
# train_fraction = 0.01 -> Accuracy on the test data after training is 0.58747935
print(train_data.shape)

train_fractions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
accuracies = [0.58747935, 0.6097013, 0.6371284,0.64840895, 0.6537764, 0.6562758, 0.65798634, 0.65859824, 0.658989]

errors = [0.12, 0.1, 0.1, 0.1, 0.09,0.09, 0.08, 0.08, 0.08]
plt.plot(train_fractions, accuracies, color = 'r')
_accs = []
_accs_ = []
for ind, acc in enumerate(accuracies):
    _accs.append(acc - errors[ind])
    _accs_.append(acc + errors[ind])
# plt.fill_between(train_fractions, _accs, _accs_, color = 'r', alpha = 0.1)

# plt.ylim(0.10, 0.80)
plt.xlabel('Fraction of the data used for training', fontsize = 14)
plt.ylabel('Accuracy on the test data', fontsize = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize =14)
plt.title('Varying the train set', fontsize =14)
plt.show()