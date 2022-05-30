from functions_plotting import plot_calibration_curve
import os 
import functions_data, functions_ordinal
import torch 

# import data 
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

# separate train, test and validation data; random seed is automatically set to 0 
train_data, validation_data, test_data = functions_ordinal.separate_train_validation_test_data(data_tensor)

n_students, n_questions = data_tensor.shape[0], data_tensor.shape[1]
max_score = int(data_tensor.max().numpy())


# import pre-trained parameters on the training data 
bs = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_bs_trained_4000iter.pth'))
bq0 = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_bq0_trained_4000iter.pth'))
rho = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_rho_trained_4000iter.pth'))

# load trained parameters 
# bs = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_bs_nofixing_prob.pth'))
# bq0 = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_bq0_nofixing_prob.pth'))
# rho = torch.load(os.path.join(os.getcwd(), 'params', '3EXAMS_rho_nofixing_prob.pth'))

# predict scores using trained parameters 
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, new_dictionary)

# prob_matrix = functions_ordinal.generate_prob_matrix_without_fixing(test_data, bs, bq0, rho, max_score)
prob_matrix = functions_ordinal.generate_prob_matrix(test_data, bs, bq0, rho, max_score, test_data_max_scores)
predicted_scores = prob_matrix.argmax(axis = 1)
true_scores = test_data[:, 2]

plot_calibration_curve(test_data, prob_matrix)
