import matplotlib.pyplot as plt 
import torch 
import os 
import numpy as np 
from matplotlib.ticker import FormatStrFormatter
from functions_test_manipulations import get_test, get_dictionary, merge_dictionaries, merge_tests
import functions_ordinal
import pickle as pkl 
import pandas as pd 


# load trained parameters 
bs_trained = torch.load(os.path.join(os.getcwd(), 'params', 'bs_trained_4000iter.pth'))
bq0_trained = torch.load(os.path.join(os.getcwd(), 'params', 'bq0_trained_4000iter.pth'))
rho_trained = torch.load(os.path.join(os.getcwd(), 'params', 'rho_trained_4000iter.pth'))

# load loss values for training, testing and validation data 
nlls_train = torch.load(os.path.join(os.getcwd(), 'params', 'nlls_train.npy'))
nlls_validation = torch.load(os.path.join(os.getcwd(), 'params', 'nlls_validation.npy'))
nlls_test = torch.load(os.path.join(os.getcwd(), 'params', 'nlls_test.npy'))


nlls_train = [x/0.8 for x in nlls_train]
nlls_test = [x/0.1 for x in nlls_test]
nlls_validation = [x/0.1 for x in nlls_validation]

plt.plot(nlls_train, label = 'train loss')
plt.plot(nlls_test, label = 'test loss')
plt.plot(nlls_validation, label  = 'valid. loss')
plt.xlabel('iteration', fontsize = 13)
plt.ylabel('loss', fontsize = 13)

plt.legend()
fig_save_loc = os.path.join(os.getcwd(), 'figures', 'ordinal_1H2H_training.png')
plt.savefig(fig_save_loc)
print('Figure saved ' + fig_save_loc)

plt.show()
# import train/test/validation data 

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

train_data, validation_data, test_data = functions_ordinal.separate_train_validation_test_data(ordinal_tensor)

# Get accuracies and predictions on the test data 
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, dict_m)
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

save_fig_location = os.path.join(os.getcwd(), 'figures', 'test_1H2H_grid.png')
plt.savefig(save_fig_location)
plt.show()

print('Accuracy on the test data after training is', functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy())
print(test_m.shape)
# data.shape = (12305, 47)
# Accuracy on the test data is 0.659 after training parameters 
