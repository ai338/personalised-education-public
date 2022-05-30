import torch
import os 
import functions_data
import functions_ordinal
import functions_plotting
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import Counter


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

# predict scores using trained parameters 
test_data_max_scores = functions_ordinal.generate_max_scores_tensor(test_data, new_dictionary)

prob_matrix = functions_ordinal.generate_prob_matrix(test_data, bs, bq0, rho, max_score, test_data_max_scores)
predicted_scores = prob_matrix.argmax(axis = 1)
true_scores = test_data[:, 2]

# functions_plotting.confusion_matrix_with_marginals(true_scores=true_scores, predicted_scores=predicted_scores)

fig, ax = plt.subplots(1, 6, constrained_layout = True)
fig.set_figheight(4), fig.set_figwidth(20)

for i in range(6): 
    df = pd.DataFrame({'trueScores': true_scores, 'predictedScores':predicted_scores, 'MaxScores':test_data_max_scores})
    df_inspect = df.loc[(df['trueScores'] == i) & (df['predictedScores'] == 0)]
    max_scores_to_inspect = df_inspect['MaxScores'].values.flatten()

    counter = Counter(max_scores_to_inspect)
    ax[i].bar(counter.keys(), height=counter.values(), color = 'lightcoral')
    # ax[i].xlabel('Maximum achieveable score on questions', fontsize = 14)
    # plt.ylabel('Count', fontsize = 14)
    ax[i].set_title(f'0 predicted, {i} true', fontsize = 14)
    ax[i].set_xlabel('max achievable score', fontsize = 14)

ax[0].set_ylabel('count of pred.', fontsize = 14)
plt.suptitle('Distribution of the maximum achievable score on the questions for which 0 is predicted', fontsize = 14)
plt.show()

