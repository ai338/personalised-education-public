import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import os 

from functions_data import binarise_using_max, import_data, shuffle_data, convert_df_to_tensor



def create_fake_tensor(n_students, n_questions):
    seed_number = 1000
    rng = torch.Generator()
    rng.manual_seed(seed_number)
    Bs = torch.normal(mean=0, std=10, size=(n_students,), requires_grad=True, generator=rng)
    Bq = torch.normal(mean=0, std=10, size=(n_questions,), requires_grad=True, generator=rng)

    Bs_matrix = Bs.repeat(n_questions, 1)
    Bs_matrix = torch.transpose(Bs_matrix, 0, 1)
    Bq_matrix = Bq.repeat(n_students, 1)

    probit = 1 / (1 + torch.exp(-Bs_matrix - Bq_matrix))
    fake_tensor = torch.bernoulli(probit)

    return fake_tensor


def separate_train_test_data(binary_data, students_separation_point, questions_separation_point):
    test = binary_data[students_separation_point:, questions_separation_point:]
    train_1 = binary_data[:students_separation_point, questions_separation_point:]
    train_2 = binary_data[students_separation_point:, :questions_separation_point]

    assert (train_2.shape == test.shape)

    return train_1, train_2, test


def zeros_ones_data(binary_data):
    bin_avg_frac = binary_data.values.sum() / binary_data.size * 100
    print('Fraction of ones in the binary dataset is ',
          "{0:0.2f}".format(bin_avg_frac), "%")


def MLE_estimator(binary_train_data):
    learning_rate = 0.0003
    n_students = binary_train_data.shape[0]
    n_questions = binary_train_data.shape[1]

    NLL = 0
    NLLs = []
    dtype = torch.FloatTensor
    seed_number = 1000
    rng = torch.Generator()
    rng.manual_seed(seed_number)
    Bs = torch.randn(n_students, requires_grad=True, generator=rng)
    Bq = torch.randn(n_questions, requires_grad=True, generator=rng)

    for iter in range(100):
        Bs_matrix = Bs.repeat(n_questions, 1)
        Bs_matrix = torch.transpose(Bs_matrix, 0, 1)
        Bq_matrix = Bq.repeat(n_students, 1)

        probit_1 = torch.log(1 / (1 + torch.exp(-Bs_matrix - Bq_matrix)))
        probit_0 = torch.log(1 / (1 + torch.exp(+Bs_matrix + Bq_matrix)))
        NLL = -torch.sum(binary_train_data * probit_1 + (1 - binary_train_data) * probit_0)

        probit_1 = torch.log(1 / (1 + torch.exp(-Bs_matrix - Bq_matrix)))
        probit_0 = torch.log(1 / (1 + torch.exp(+Bs_matrix + Bq_matrix)))
        NLL = -torch.sum(binary_train_data * probit_1 + (1 - binary_train_data) * probit_0)

        NLL.backward()

        Bs.data -= learning_rate * Bs.grad.data
        Bq.data -= learning_rate * Bq.grad.data

        # Manually zero the gradients after updating weights
        Bs.grad.data.zero_()
        Bq.grad.data.zero_()

        NLLs.append(NLL.detach().numpy())

    plt.plot(NLLs)
    plt.xlabel('Iteration of the maximum likelihood algorithm')
    plt.ylabel('Negative log likelihood')
    plt.show()
    return Bs, Bq


def test_MLE(test_data, Bs_est, Bq_est):
    n_students = test_data.shape[0]
    n_questions = test_data.shape[1]
    predictions = torch.zeros_like(test_data).t()

    predictions += Bq_est.detach().repeat(n_students, 1).t()
    predictions += Bs_est.detach().repeat(n_questions, 1)

    predictions = predictions.t()

    m = nn.Sigmoid()

    sigmoid = m(predictions)
    print(sigmoid)
    predictions = torch.bernoulli(sigmoid)

    accuracy = torch.count_nonzero(predictions == test_data)
    plt.hist(predictions)
    plt.show()
    plt.hist(test_data)
    plt.show()
    return accuracy / (n_students * n_questions)


def fake_data_experiments():
    n_s = 50000
    n_q = 24
    students_separation_point = n_s // 2
    questions_separation_point = n_q // 2

    seed_number = 1000
    rng = torch.Generator()
    rng.manual_seed(seed_number)
    Bs_true = torch.normal(mean=0, std=10, size=(n_s,), requires_grad=True, generator=rng)
    Bq_true = torch.normal(mean=0, std=10, size=(n_q,), requires_grad=True, generator=rng)

    fake_df_tensor = create_fake_tensor(n_s, n_q)

    train_1_fake, train_2_fake, test_fake = separate_train_test_data(fake_df_tensor, students_separation_point,
                                                                     questions_separation_point)

    Bs_estimated_1, Bq_estimated_1 = MLE_estimator(train_1_fake)

    Bs_estimated_2, Bq_estimated_2 = MLE_estimator(train_2_fake)

    Bs_estimated = Bs_estimated_1 + Bs_estimated_2
    Bq_estimated = Bq_estimated_2 + Bq_estimated_1

    # acc = test_MLE(test_fake, Bs_estimated_2, Bq_estimated_1)
    # print(acc)
    # plt.plot(loss_res_2q)
    # plt.title('MSE loss iterations of gradient descent')
    # plt.show()


def real_data_experiments(binarised_data):

    df = convert_df_to_tensor(binarised_data)

    # df = df[:500, :]
    n_s = df.shape[0]
    n_q = df.shape[1]
    students_separation_point = n_s // 2
    questions_separation_point = n_q // 2

    train_1, train_2, test = separate_train_test_data(df, students_separation_point,
                                                      questions_separation_point)

    Bs_estimated_1, Bq_estimated_1 = MLE_estimator(train_1)
    Bs_estimated_2, Bq_estimated_2 = MLE_estimator(train_2)

    Bs_estimated = Bs_estimated_1 + Bs_estimated_2
    Bq_estimated = Bq_estimated_2 + Bq_estimated_1

    acc = test_MLE(test, Bs_estimated_2, Bq_estimated_1)
    print(acc)


data_file = os.path.join(os.getcwd(), 'data', '9to1_2017_GCSE_1H.csv')
df, q_info = import_data(data_file)
shuffle_data(df) 
df = binarise_using_max(df, q_info)

real_data_experiments(df)