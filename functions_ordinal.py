import torch
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import seaborn as sns 
import pandas as pd 

sigmoid = torch.nn.Sigmoid()


def correct_predictions_test_set(predicted_scores, true_scores):
    correct = (true_scores == predicted_scores).int()
    n = correct.shape[0]*correct.shape[1]
    percentage_correct = correct.sum()/n
    return percentage_correct 


def generate_single_question_tensor(bq0_gen, rho_gen, question_index):
    bq0 = bq0_gen[question_index].clone()
    
    bq_tensor = rho_gen[question_index].clone() 
    bq_tensor = torch.exp(bq_tensor)
    bq_tensor = torch.cat((bq0.reshape(1), bq_tensor))

    return torch.cumsum(bq_tensor, dim=0)


def generate_and_plot_synthetic_data(n_students, n_questions, r1, r2, max_score):
    sigmoid = torch.nn.Sigmoid()
    torch.manual_seed(0)

    bs_generator = torch.randn(n_students)
    # bs_generator = torch.normal(0, 0.001, size = (n_students, ))
    # r1 = -2 
    # r2 = 2
    bq0_generator = (r1 - r2)*torch.rand(n_questions) + r2  
    rho_generator = torch.normal(0, 0.1, size = (n_questions, max_score)) # rho1, rho2, rho3  

    q_parameters = []
    
    for question_index in range(n_questions):
        question_tensor = generate_single_question_tensor(bq0_gen=bq0_generator, rho_gen = rho_generator, question_index=question_index)
        q_parameters.append(question_tensor)


    q_parameters = torch.stack(q_parameters)

    s_parameters = bs_generator 

    # generate complete_matrix_scores with given q_parameters and s_parameters
    concat_scores  = torch.tensor([])

    for q in range(n_questions):
        Kq = max_score + 1 # number of parameters per question 
        q_vector = q_parameters[q]
        # print(q_vector)
        
        # step 1 
        q_matrix = q_vector.repeat((n_students, 1))

        # step 2 
        s_matrix = s_parameters.repeat_interleave(Kq, dim=0).reshape((n_students, Kq))

        # step 3 
        prediction_matrix = q_matrix + s_matrix 

        # step 4 
        prediction_matrix_1 = sigmoid(prediction_matrix)

        # step 5 
        prediction_matrix_padded_with_zeros = F.pad(input=prediction_matrix_1, pad=(1, 0, 0, 0), mode='constant', value=0)

        # step 6 
        prediction_matrix_2 = prediction_matrix_padded_with_zeros[:, :-1]

        # step 7 
        prediction_matrix_probabilities = prediction_matrix_1 - prediction_matrix_2
        
        # step 7' setting the last probability to be 1 - sum of the rest 
        diff = torch.ones(n_students) - prediction_matrix_probabilities.sum(axis=1)
        prediction_matrix_probabilities[:, -1] += diff 


        questions_scores = torch.multinomial(prediction_matrix_probabilities, 1, replacement=True)

        concat_scores = torch.cat((concat_scores, questions_scores), axis=1)
    
    # torch data to np data in order to plot it in seaborn 
    syntetic_data_np = concat_scores.detach().numpy()
    syntetic_data_np = syntetic_data_np.flatten()

    # sanity check how it prints 
    # print(syntetic_data_np)


    sns.distplot(syntetic_data_np, label='r1=' + str(r1) + ' r2=' + str(r2))
    #plt.hist(syntetic_data_np, bins = 10)
    plt.xlabel('score')
    plt.ylabel('distribution of scores')
    plt.title('Distribution of synthetic scores with 24 questions and 50000 students')
    plt.legend()
    plt.show()

    return concat_scores

def dataframe_to_triplets(ordinal_data):
    """
    Converts dataframe (rows are questions; columns are students) to triplets in the form 
    [studentID, questionID, score] for each student-question pair 
    """
    n_students, n_questions = ordinal_data.shape[0], ordinal_data.shape[1]

    ordinal_data = ordinal_data.reshape(-1)

    student_id = torch.arange(1, n_students + 1)
    student_id = student_id.repeat_interleave(n_questions)

    question_id = torch.arange(1, n_questions + 1)
    question_id = question_id.repeat(n_students)

    data = torch.stack((student_id, question_id, ordinal_data), dim=1)
    return data 

def separate_train_validation_test_data(ordinal_data, random_seed = 0):
    """
    Separate data: train/validation/test split to be 80/10/10
    ordinal_data needs to be tensor! 
    Data is returned in the form of triples, where each triple is (student_id, question_id, score)

    """    
    n_students = ordinal_data.shape[0]
    n_questions = ordinal_data.shape[1]

    ordinal_data = ordinal_data.reshape(-1)

    student_id = torch.arange(1, n_students + 1)
    student_id = student_id.repeat_interleave(n_questions)

    question_id = torch.arange(1, n_questions + 1)
    question_id = question_id.repeat(n_students)

    data = torch.zeros([n_students*n_questions, 3])
    data = torch.stack((student_id, question_id, ordinal_data), dim=1)

    torch.manual_seed(random_seed)
    data = data[torch.randperm(data.size()[0])]
    
    n_train = int(data.shape[0]*0.8)
    n_validation = int(data.shape[0]*0.1)
    n_test = data.shape[0]-(n_train + n_validation)

    train_data = data[:n_train, :]
    validation_data = data[n_train:n_train+n_validation, :]
    test_data = data[n_train+n_validation:, :]

    train_data[:, 0] = train_data[:, 0].int()
    train_data[:, 1] = train_data[:, 1].int()

    validation_data[:, 0] = validation_data[:, 0].int()
    validation_data[:, 1] = validation_data[:, 1].int()

    test_data[:, 0] = test_data[:, 0].int()
    test_data[:, 1] = test_data[:, 1].int()


    return train_data, validation_data, test_data


def plot_distribution_of_scores(scores_data, title=None):
    # torch data to np data in order to plot it in seaborn 
    plot_scores_data = scores_data.detach().numpy()
    sns.distplot(plot_scores_data)
    #plt.hist(syntetic_data_np, bins = 10)
    plt.xlabel('score')
    plt.ylabel('distribution of scores')
    if title:
        plt.title(title)
    plt.show()



def full_accuracy(predicted_scores, true_scores):
    n = predicted_scores.shape[0]
    corr = (predicted_scores == true_scores).int().sum()
    return corr/n 

def one_point_off_accuracy(predicted_scores, true_scores):
    n = predicted_scores.shape[0]
    corr = 0 
    for i in range(n):
        if torch.abs(predicted_scores[i] - true_scores[i]) <= 1:
            corr += 1 
    
    return corr / n 

def nll(data, params, max_score, tensor_max_scores):
    bs_tensor = params[0]
    bq0_tensor = params[1]
    rho_tensor = params[2]

    probabilities_matrix = generate_prob_matrix(data, bs_tensor, bq0_tensor, rho_tensor, max_score, tensor_max_scores)

    scores_vector = (data[:, 2]).int()
    score_indices = scores_vector.unsqueeze(1)
    score_indices = score_indices.long()

    selected_probabilities = probabilities_matrix.gather(1, score_indices).flatten()
    log_likelihood = torch.log(selected_probabilities).sum()

    negative_log_likelihood = -log_likelihood
    return negative_log_likelihood 

def nll_no_fixing(data, params, max_score):
    bs_tensor = params[0]
    bq0_tensor = params[1]
    rho_tensor = params[2]

    probabilities_matrix = generate_prob_matrix_without_fixing(data, bs_tensor, bq0_tensor, rho_tensor, max_score)

    scores_vector = (data[:, 2]).int()
    score_indices = scores_vector.unsqueeze(1)
    score_indices = score_indices.long()

    selected_probabilities = probabilities_matrix.gather(1, score_indices).flatten()
    log_likelihood = torch.log(selected_probabilities).sum()

    negative_log_likelihood = -log_likelihood
    return negative_log_likelihood 



def generate_prob_matrix(data, bs_tensor, bq0_tensor, rho_tensor, max_score, tensor_max_scores):
    n = data.shape[0] # number of datapoints 

    # select student and question indices in the triplets data 
    s_indices = (data[:, 0] - 1).int() # substract one because in the table indices of students and questions start with 1 
    q_indices = (data[:, 1] - 1).int()

    bs_vector = torch.index_select(bs_tensor, 0, s_indices)
    bq0_vector = torch.index_select(bq0_tensor, 0, q_indices)
    rho_vector = torch.index_select(rho_tensor, 0, q_indices)

    rho_exp_vector = torch.exp(rho_vector)

    bq_matrix = torch.hstack([bq0_vector.unsqueeze(1), rho_exp_vector])
    bq_matrix = bq_matrix.cumsum(axis=1)

    bs_matrix = bs_vector.tile((max_score,1))
    bs_matrix = torch.transpose(bs_matrix, 0, 1)

    bs_bq_matrix = bs_matrix + bq_matrix
    bs_bq_matrix_sigmoid = sigmoid(bs_bq_matrix)

    ones = torch.ones((n, 1))
    zeros = torch.zeros((n,1))
    # append 1 to the right 
    sigma_higher = torch.cat((bs_bq_matrix_sigmoid, ones),1)
    # append 0 to the left 
    sigma_lower = torch.cat((zeros, bs_bq_matrix_sigmoid),1)

    probabilities_matrix = sigma_higher - sigma_lower

    fix_probabilites(probabilities_matrix, tensor_max_scores)

    return probabilities_matrix

def generate_prob_matrix_without_fixing(data, bs_tensor, bq0_tensor, rho_tensor, max_score):
    n = data.shape[0] # number of datapoints 

    # select student and question indices in the triplets data 
    s_indices = (data[:, 0] - 1).int() # substract one because in the table indices of students and questions start with 1 
    q_indices = (data[:, 1] - 1).int()

    bs_vector = torch.index_select(bs_tensor, 0, s_indices)
    bq0_vector = torch.index_select(bq0_tensor, 0, q_indices)
    rho_vector = torch.index_select(rho_tensor, 0, q_indices)

    rho_exp_vector = torch.exp(rho_vector)

    bq_matrix = torch.hstack([bq0_vector.unsqueeze(1), rho_exp_vector])
    bq_matrix = bq_matrix.cumsum(axis=1)

    bs_matrix = bs_vector.tile((max_score,1))
    bs_matrix = torch.transpose(bs_matrix, 0, 1)

    bs_bq_matrix = bs_matrix + bq_matrix
    bs_bq_matrix_sigmoid = sigmoid(bs_bq_matrix)

    ones = torch.ones((n, 1))
    zeros = torch.zeros((n,1))
    # append 1 to the right 
    sigma_higher = torch.cat((bs_bq_matrix_sigmoid, ones),1)
    # append 0 to the left 
    sigma_lower = torch.cat((zeros, bs_bq_matrix_sigmoid),1)

    probabilities_matrix = sigma_higher - sigma_lower


    return probabilities_matrix


def predict_scores(data, bs_tensor, bq0_tensor, rho_tensor, max_score, tensor_max_scores):
     
    probabilities_matrix = generate_prob_matrix(data, bs_tensor, bq0_tensor, rho_tensor, max_score, tensor_max_scores)

    questions_scores_predicted = torch.multinomial(probabilities_matrix, 1, replacement=True).flatten()

    return questions_scores_predicted


def find_max_score(row, questions_info):
    q_id = row['questionID']
    return questions_info['q' + str(int(q_id))]['Max']

def generate_max_scores_tensor(data, questions_info):
    dataframe = pd.DataFrame(data).astype("float")
    dataframe = dataframe.rename(columns={0: "studentID", 1: "questionID", 2: "score"})

    dataframe['maxScore'] = dataframe.apply(lambda row: find_max_score(row, questions_info), axis=1)

    max_scores_tensor = torch.tensor(dataframe['maxScore'].values)
    
    return max_scores_tensor

def fix_probabilites(probability_matrix, tensor_max_scores):
    for maximum_score in range(1, 6):
        indices = (tensor_max_scores == maximum_score).nonzero().flatten()
        probability_matrix[indices, maximum_score + 1:] = 0
        probability_matrix[indices, maximum_score] = 1 - probability_matrix[indices, :maximum_score].sum(axis=1)


def custom_separate_train_validation_test_data(ordinal_data, train_fraction = 0.8, valid_fraction = 0.1, test_fraction = 0.1, random_seed = 0):
    """
    Separate data: train/validation/test split to be 80/10/10
    ordinal_data needs to be tensor! 
    Data is returned in the form of triples, where each triple is (student_id, question_id, score)

    """    
    n_students = ordinal_data.shape[0]
    n_questions = ordinal_data.shape[1]

    ordinal_data = ordinal_data.reshape(-1)

    student_id = torch.arange(1, n_students + 1)
    student_id = student_id.repeat_interleave(n_questions)

    question_id = torch.arange(1, n_questions + 1)
    question_id = question_id.repeat(n_students)

    data = torch.zeros([n_students*n_questions, 3])
    data = torch.stack((student_id, question_id, ordinal_data), dim=1)

    torch.manual_seed(random_seed)
    data = data[torch.randperm(data.size()[0])]
    
    n_train = int(data.shape[0]*train_fraction)
    n_validation = int(data.shape[0]*valid_fraction)
    n_test = int(data.shape[0]*test_fraction)

    train_data = data[:n_train, :]
    validation_data = data[n_train:n_train+n_validation, :]
    test_data = data[-n_test:, :]

    train_data[:, 0] = train_data[:, 0].int()
    train_data[:, 1] = train_data[:, 1].int()

    validation_data[:, 0] = validation_data[:, 0].int()
    validation_data[:, 1] = validation_data[:, 1].int()

    test_data[:, 0] = test_data[:, 0].int()
    test_data[:, 1] = test_data[:, 1].int()


    return train_data, validation_data, test_data