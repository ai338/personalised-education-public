from functions_plotting import plot_calibration_curve
import os 
import functions_data, functions_ordinal
import torch 
from torch import optim 


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

# initialise parameters 
torch.manual_seed(100)
bs = torch.randn(n_students, requires_grad=True)
bq0 = torch.randn([n_questions, ], requires_grad=True)
rho = torch.normal(0, 0.1, size = (n_questions, max_score - 1), requires_grad=True)

nlls_train, nlls_valid, nlls_test = [], [], []

# generate max_scores tensor 
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, new_dictionary)
train_data_max_scores = functions_ordinal.generate_max_scores_tensor(train_data, new_dictionary)
validation_data_max_scores = functions_ordinal.generate_max_scores_tensor(validation_data, new_dictionary)

# train 
iterations = 4000 
params = [bs, bq0, rho]

opt = optim.SGD(params, lr = 0.0001)

for iter in range(iterations):
    loss = functions_ordinal.nll_no_fixing(train_data, params, max_score)    
    opt.zero_grad()
    loss.backward()
    opt.step()
    nlls_train.append(loss.detach().numpy())

    loss_validation = functions_ordinal.nll_no_fixing(validation_data, params, max_score)
    loss_test = functions_ordinal.nll_no_fixing(test_data, params, max_score)

    nlls_valid.append(loss_validation.detach().numpy())
    nlls_test.append(loss_test.detach().numpy())

    if iter % 100 == 0:
        print(iter, loss)


# save trained parameters 
torch.save(bs, os.path.join(os.getcwd(), 'params', '3EXAMS_bs_nofixing_prob.pth'))
torch.save(bq0, os.path.join(os.getcwd(), 'params', '3EXAMS_bq0_nofixing_prob.pth'))
torch.save(rho, os.path.join(os.getcwd(), 'params', '3EXAMS_rho_nofixing_prob.pth'))

# save array of losses 
torch.save(nlls_train, os.path.join(os.getcwd(), 'params', '3EXAMS_nlls_train_nofixing_prob.npy'))
torch.save(nlls_valid, os.path.join(os.getcwd(), 'params', '3EXAMS_nlls_validation_nofixing_prob.npy'))
torch.save(nlls_test, os.path.join(os.getcwd(), 'params', '3EXAMS_nlls_test_nofixing_prob.npy'))

print('Done with executing the script')
