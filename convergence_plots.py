from shutil import which
import matplotlib.pyplot as plt 
import torch 
import os 
import numpy as np 
from operator import add 


seeds = np.arange(1000, 1010, 1)
fig, ax = plt.subplots(1, 2)
fig.set_figheight(6)
fig.set_figwidth(15)

for random_seed in seeds: 
    # load bs trajectories for the random seed 
    seed_path_1 = 'bs_conv1_init_seed_{}.pth'.format(random_seed)
    seed_path_2 = 'bs_conv2_init_seed_{}.pth'.format(random_seed)
    seed_path_3 = 'bs_conv3_init_seed_{}.pth'.format(random_seed)
    bs_1 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_1))
    bs_2 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_2))
    bs_3 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_3))

    # load bq0 trajectories for the random seed 
    seed_path_1 = 'bq0_conv1_init_seed_{}.pth'.format(random_seed)
    seed_path_2 = 'bq0_conv2_init_seed_{}.pth'.format(random_seed)
    seed_path_3 = 'bq0_conv3_init_seed_{}.pth'.format(random_seed)

    bq0_1 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_1))
    bq0_2 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_2))
    bq0_3 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_3))

    if random_seed == seeds[-1]:
        ax[0].plot(bs_1, color='r', label = r'$b_{s=10}$')
        ax[0].plot(bs_2, color = 'b', label = r'$b_{s=15}$')
        ax[0].plot(bs_3, color = 'grey', label = r'$b_{s=20}$')
        ax[1].plot(bq0_1, color = 'g', label = r'$b_{q0=10}$')
        ax[1].plot(bq0_2, color = 'orange', label = r'$b_{q0=16}$')
        ax[1].plot(bq0_3, color = 'purple', label = r'$b_{q0=25}$')
        # ax[2].plot(list(map(add, bs_1, bq0_1)), color = 'lightcoral', label = r'$b_{s=10} + b_{q0=10}$')
        # ax[2].plot(list(map(add, bs_2, bq0_2)), color = 'lightgreen', label = r'$b_{s=15} + b_{q0=16}$')
        # ax[2].plot(list(map(add, bs_2, bq0_3)), color = 'lightblue', label = r'$b_{s=15} + b_{q0=25}$')

    else:
        ax[0].plot(bs_1, color='r')
        ax[0].plot(bs_2, color = 'b')
        ax[0].plot(bs_3, color = 'grey')
        ax[1].plot(bq0_1, color = 'g')
        ax[1].plot(bq0_2, color = 'orange')
        ax[1].plot(bq0_3, color = 'purple')
        # ax[2].plot(list(map(add, bs_1, bq0_1)), color = 'lightcoral')
        # ax[2].plot(list(map(add, bs_2, bq0_2)), color = 'lightgreen')
        # ax[2].plot(list(map(add, bs_2, bq0_3)), color = 'lightblue')


ax[1].set_xlabel('iteration', fontsize = 15)
ax[0].set_xlabel('iteration', fontsize = 15)
# ax[2].set_xlabel('iteration', fontsize = 15)


ax[0].set_ylabel(r'Student parameter $b_s$', fontsize = 15)
ax[1].set_ylabel(r'Question parameter $b_{q0}$', fontsize = 15)
# ax[2].set_ylabel(r'$b_s + b_{q0}$', fontsize = 15)


ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
# ax[2].tick_params(axis='both', which='major', labelsize=15)

fig.legend(fontsize = 15)
# fig.suptitle('Convergence of student and question parameters for different random seeds used for initialisation', fontsize = 14)
# plt.savefig(os.path.join(os.getcwd(), 'report_figures', 'convergence_parameters.png'))
plt.show()

# PLOT CONVERGENCE OF SUM OF THE PARAMETERS 

for random_seed in seeds: 
    # load bs trajectories for the random seed 
    seed_path_1 = 'bs_conv1_init_seed_{}.pth'.format(random_seed)
    seed_path_2 = 'bs_conv2_init_seed_{}.pth'.format(random_seed)
    seed_path_3 = 'bs_conv3_init_seed_{}.pth'.format(random_seed)
    bs_1 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_1))
    bs_2 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_2))
    bs_3 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_3))

    # load bq0 trajectories for the random seed 
    seed_path_1 = 'bq0_conv1_init_seed_{}.pth'.format(random_seed)
    seed_path_2 = 'bq0_conv2_init_seed_{}.pth'.format(random_seed)
    seed_path_3 = 'bq0_conv3_init_seed_{}.pth'.format(random_seed)

    bq0_1 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_1))
    bq0_2 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_2))
    bq0_3 = torch.load(os.path.join(os.getcwd(), 'seed_params', seed_path_3))

    if random_seed == seeds[-1]:
        plt.plot(list(map(add, bs_1, bq0_1)), color = 'lightcoral', label = r'$b_{s=10} + b_{q0=10}$')
        plt.plot(list(map(add, bs_2, bq0_2)), color = 'lightgreen', label = r'$b_{s=15} + b_{q0=16}$')
        plt.plot(list(map(add, bs_2, bq0_3)), color = 'lightblue', label = r'$b_{s=15} + b_{q0=25}$')

    else:
        plt.plot(list(map(add, bs_1, bq0_1)), color = 'lightcoral')
        plt.plot(list(map(add, bs_2, bq0_2)), color = 'lightgreen')
        plt.plot(list(map(add, bs_2, bq0_3)), color = 'lightblue')

plt.xlabel('iteration', fontsize = 15)
plt.tick_params(axis='both', which='major', labelsize = 15)
plt.ylabel(r'$b_s + b_{q0}$', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()
