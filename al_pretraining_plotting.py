import torch 
import os 
import matplotlib.pyplot as plt 
from functions_data import convert_dataframe_to_triples, load_data_and_dictionary
from functions_ordinal import full_accuracy, generate_prob_matrix
import functions_ordinal
import functions_plotting

#load pretrained parameters 
bs = torch.load(os.path.join(os.getcwd(), 'al_params', 'bs_pretrained.pth'))
bq0 = torch.load(os.path.join(os.getcwd(), 'al_params', 'bq0_pretrained.pth'))
rho = torch.load(os.path.join(os.getcwd(), 'al_params', 'rho_pretrained.pth'))

# save array of losses 
train_loss = torch.load(os.path.join(os.getcwd(), 'al_params', 'nlls_train_pretrained.npy'))
valid_loss = torch.load(os.path.join(os.getcwd(), 'al_params', 'nlls_valid_pretrained.npy'))
test_loss = torch.load(os.path.join(os.getcwd(), 'al_params', 'nlls_test_pretrained.npy'))

data_path = os.path.join(os.getcwd(), 'data', 'three_exams_cleaned.csv')
dict_path = os.path.join(os.getcwd(), 'data', 'questions_info_dict.p')
data, questions_info = load_data_and_dictionary(data_path, dict_path)

n_students, n_questions = data.shape[0], data.shape[1]

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

max_score = int(data.values.max())


training_students = 13563
pooling_students = data.shape[0] - training_students

training_dataframe = data[:training_students]
pooling_dataframe = data[training_students:]

torch.random.manual_seed(100)

# Shuffle training data and separate train test and validation data within the training data for active learning 
data_training_triplets = convert_dataframe_to_triples(training_dataframe)
shuffled_train_pool_data = data_training_triplets[torch.randperm(data_training_triplets.size()[0])]
n_datapoints = shuffled_train_pool_data.shape[0]

# 80/10/10 train/test/validation split
train_data_tp = shuffled_train_pool_data[:int(0.8*n_datapoints)]
test_data_tp = shuffled_train_pool_data[int(0.8*n_datapoints):int(0.9*n_datapoints)]
validation_data_tp = shuffled_train_pool_data[int(0.9*n_datapoints):]


# max scores for train, test and validation in the training pool 
tp_train_max = functions_ordinal.generate_max_scores_tensor(train_data_tp, new_dictionary)
tp_valid_max = functions_ordinal.generate_max_scores_tensor(validation_data_tp, new_dictionary)
tp_test_max = functions_ordinal.generate_max_scores_tensor(test_data_tp, new_dictionary)

train_loss = [x / train_data_tp.shape[0] for x in train_loss]
valid_loss = [x / validation_data_tp.shape[0] for x in valid_loss]
test_loss = [x / test_data_tp.shape[0] for x in test_loss]

plt.rcParams["figure.figsize"] = (8,5)

plt.plot(train_loss, label = 'train nll')
plt.plot(valid_loss, label = 'validation nll')
plt.plot(test_loss, label = 'test nll')
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlabel('iteration', fontsize = 16)
plt.ylabel('negative log-likelihood (loss)', fontsize = 16)
plt.title('Active learning - pretraining loss', fontsize = 16)
plt.legend(fontsize = 14)
plt.show()


prob_matrix = generate_prob_matrix(test_data_tp, bs, bq0, rho, max_score, tp_test_max)
predicted_scores = prob_matrix.argmax(axis = 1)
true_scores = test_data_tp[:, 2]

functions_plotting.plot_calibration_curve(test_data_tp, prob_matrix, title='AL pretraining calibration curve', figsize=(8,5))

print('Accuracy on the test data after training is', full_accuracy(true_scores, predicted_scores).numpy())