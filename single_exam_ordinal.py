import os 
from functions_data import shuffle_data, import_data
import torch 
import functions_ordinal
from torch import optim


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
torch.save(bs, os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_bs_trained_4000iter.pth'))
torch.save(bq0, os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_bq0_trained_4000iter.pth'))
torch.save(rho, os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_rho_trained_4000iter.pth'))

# save array of losses 
torch.save(nlls_train, os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_nlls_train.npy'))
torch.save(nlls_validation, os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_nlls_validation.npy'))
torch.save(nlls_test, os.path.join(os.getcwd(), 'params', 'single_exam_ordinal_nlls_test.npy'))

print('Done with executing the script')



