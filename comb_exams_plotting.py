import matplotlib.pyplot as plt 
import torch 
import os 
import numpy as np 
from matplotlib.ticker import FormatStrFormatter
from functions_test_manipulations import get_merged_test_and_dict, get_test, get_dictionary, merge_dictionaries, merge_tests
import functions_ordinal
import pickle as pkl 
import pandas as pd 

# load trained parameters 
bs_trained = torch.load(os.path.join(os.getcwd(), 'params', 'bs_trained_4000iter_8T.pth'))
bq0_trained = torch.load(os.path.join(os.getcwd(), 'params', 'bq0_trained_4000iter_8T.pth'))
rho_trained = torch.load(os.path.join(os.getcwd(), 'params', 'rho_trained_4000iter_8T.pth'))

# load loss values for training, testing and validation data 
nlls_train = torch.load(os.path.join(os.getcwd(), 'params', 'nlls_train_8T.npy'))
nlls_validation = torch.load(os.path.join(os.getcwd(), 'params', 'nlls_validation_8T.npy'))
nlls_test = torch.load(os.path.join(os.getcwd(), 'params', 'nlls_test_8T.npy'))


nlls_train = [x/0.8 for x in nlls_train]
nlls_test = [x/0.1 for x in nlls_test]
nlls_validation = [x/0.1 for x in nlls_validation]

plt.plot(nlls_train, label = 'train loss')
plt.plot(nlls_test, label = 'test loss')
plt.plot(nlls_validation, label  = 'valid. loss')
plt.xlabel('iteration', fontsize = 13)
plt.ylabel('loss', fontsize = 13)

plt.legend()
fig_save_loc = os.path.join(os.getcwd(), 'figures', 'ordinal_1H2H_training_8T.png')
plt.savefig(fig_save_loc)
print('Figure saved ' + fig_save_loc)

plt.show()


# LOAD THREE EXAMS AND TEST/TRAIN/VALIDATION DATA

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

# TEST GRID PREDICTIONS 
# Get accuracies and predictions on the test data 
prob_matrix = functions_ordinal.generate_prob_matrix(test_data, bs_trained, bq0_trained, rho_trained, max_score, test_data_max_scores)

predicted_scores_thresholding = prob_matrix.argmax(axis = 1)
predicted_scores = predicted_scores_thresholding


true_scores = test_data[:, 2]

dim = 15 
pred_mat = predicted_scores[0:0+dim*dim].reshape((dim,dim))
true_mat = true_scores[0:0+dim*dim].reshape((dim,dim))

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

save_fig_location = os.path.join(os.getcwd(), 'figures', 'test_grid_8T.png')
plt.savefig(save_fig_location)
plt.show()

print('Accuracy on the test data after training is', functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy())
print(df_merged.shape)


# Accuracy on the test data in 2017 is 0.656 after training parameters; dataset size = (9247, 70)

# Accuracy on the test data in Nov2018 batch of tests is 0.681 after training parameters; dataset size = (12963, 88) = 1140744 datapoint pairs 

# Accuracy on the test data in Summer2019 is 0.666 after training parameters; dataset size = (17085, 58) = 990930 datapoint pairs 

accuracies = [65.6, 66.6, 68.1]
data_size = [9247*70, 990930, 12963 * 88] 
plt.plot(data_size, accuracies)