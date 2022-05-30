import torch 
import os
import matplotlib.pyplot as plt 
import numpy as np 

accs_entropy = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_accs_long_training.pth'))
loss_entropy = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_loss_long_training.pth'))

all_random_accs = []
random_seeds = [0, 1001, 1002, 1003, 1004]
for random_seed in random_seeds: 
    if random_seed != 0:
        accs_random = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_accs_long_training.pth'))
        loss_random = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_loss_long_training.pth'))
    else: 
        accs_random = torch.load(os.path.join(os.getcwd(), 'al_params', 'random_pooling_accs_long_training.pth'))
        loss_random = torch.load(os.path.join(os.getcwd(), 'al_params', 'random_pooling_loss_long_training.pth'))
    all_random_accs.append(np.array(accs_random))


all_random_accs = torch.tensor(np.array(all_random_accs))
variance = torch.var(all_random_accs, dim = 0)
random_mean = torch.mean(all_random_accs, dim = 0)

number_of_questions_revealed = np.arange(0, len(accs_entropy)) # x-label, will be needed for plotting the y filling 

plt.plot(number_of_questions_revealed, random_mean, label = 'random', color = 'r')
plt.plot(number_of_questions_revealed, accs_entropy, label = 'entropy', color = 'g')
plt.fill_between(number_of_questions_revealed, random_mean - variance, random_mean + variance)
plt.title('Accuracy on the test set', fontsize = 15)
plt.xlabel('number of questions revealed', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

# plt.plot(loss_random, label = 'random', color='r')
# plt.plot(loss_entropy, label = 'entropy', color = 'g')
# plt.title('Loss in the pooling phase of AL', fontsize = 15)
# plt.xlabel('AL training iteration', fontsize = 15)
# plt.legend(fontsize = 15)
# plt.show()

