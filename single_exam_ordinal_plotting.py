import os 
from functions_data import shuffle_data, import_data
import torch 
import functions_ordinal
from torch import optim
import matplotlib.pyplot as plt 

data_file = os.path.join(os.getcwd(), 'data', '9to1_2017_GCSE_1H.csv')
df, q_info = import_data(data_file)
shuffle_data(df) 



ordinal_data = df.astype('float')
ordinal_tensor = torch.tensor(ordinal_data.values)

n_students, n_questions = ordinal_tensor.shape[0], ordinal_tensor.shape[1]
max_score = int(ordinal_tensor.max().numpy())

train_data, validation_data, test_data = functions_ordinal.separate_train_validation_test_data(ordinal_tensor, random_seed = 100)

train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, q_info)
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, q_info)
validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, q_info)

# load trained parameters 
bs = torch.load(os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_bs_trained_4000iter.pth'))
bq0 = torch.load(os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_bq0_trained_4000iter.pth'))
rho = torch.load(os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_rho_trained_4000iter.pth'))

# load saved arrays of losses 
nlls_train = torch.load(os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_nlls_train.npy'))
nlls_validation = torch.load(os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_nlls_validation.npy'))
nlls_test = torch.load(os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_nlls_test.npy'))



nlls_train = [x/0.7 for x in nlls_train]
nlls_test = [x/0.1 for x in nlls_test]
nlls_validation = [x/0.2 for x in nlls_validation]

plt.plot(nlls_train, label = 'train loss')
plt.plot(nlls_test, label = 'test loss')
plt.plot(nlls_validation, label  = 'valid. loss')
plt.xlabel('iteration', fontsize = 13)
plt.ylabel('loss', fontsize = 13)
plt.show()


prob_matrix = functions_ordinal.generate_prob_matrix(test_data, bs, bq0, rho, max_score, test_data_max_scores)

predicted_scores_thresholding = prob_matrix.argmax(axis = 1)
predicted_scores = predicted_scores_thresholding


true_scores = test_data[:, 2]


print('Accuracy on the test data after training is', functions_ordinal.full_accuracy(true_scores, predicted_scores).numpy())
print('Up to a point accuracy on the test data after training is', functions_ordinal.one_point_off_accuracy(true_scores, predicted_scores))
