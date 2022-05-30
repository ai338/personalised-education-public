import torch 
import os
import matplotlib.pyplot as plt 


accs_entropy = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_accs_long_training.pth'))
loss_entropy = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_loss_long_training.pth'))

random_seed = 1004
accs_random = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_accs_long_training.pth'))
loss_random = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_loss_long_training.pth'))



plt.plot(accs_random, label = 'random', color = 'r')
plt.plot(accs_entropy, label = 'entropy', color = 'g')
plt.title('Accuracy on the test set', fontsize = 15)
plt.xlabel('number of questions revealed', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

plt.plot(loss_random, label = 'random', color='r')
plt.plot(loss_entropy, label = 'entropy', color = 'g')
plt.title('Loss in the pooling phase of AL', fontsize = 15)
plt.xlabel('AL training iteration', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()



bs_random_1 = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs1.pth'))
bs_random_2 = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs2.pth'))
bs_random_3 = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs3.pth'))
bs_random_4 = torch.load(os.path.join(os.getcwd(), 'al_params', f'random_seed_{random_seed}_pooling_bs4.pth'))

bs_entropy_1 = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_bs1.pth'))
bs_entropy_2 = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_bs2.pth'))
bs_entropy_3 = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_bs3.pth'))
bs_entropy_4 = torch.load(os.path.join(os.getcwd(), 'al_params', 'entropy_pooling_bs4.pth'))


plt.plot(bs_random_1, linestyle = '--', label = r'random $b_{s=1}$', color = 'r')
plt.plot(bs_random_2, linestyle = '--', label = r'random $b_{s=2}$', color = 'g')
plt.plot(bs_random_3, linestyle = '--', label = r'random $b_{s=3}$', color = 'b')
plt.plot(bs_random_4, linestyle = '--', label = r'random $b_{s=4}$', color = 'orange')

plt.plot(bs_entropy_1, linestyle ='-', label = r'entropy $b_{s=1}$', color = 'r')
plt.plot(bs_entropy_2, linestyle = '-', label = r'entropy $b_{s=2}$', color = 'g')
plt.plot(bs_entropy_3, linestyle = '-', label = r'entropy $b_{s=3}$', color='b')
plt.plot(bs_entropy_4, linestyle = '-', label = r'entropy $b_{s=4}$', color = 'orange')

plt.xlabel('number of questions revealed per student', fontsize = 14)
plt.ylabel(r'$b_{s}$', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.legend()
plt.show()
