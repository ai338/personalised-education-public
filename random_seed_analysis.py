from audioop import avg
import torch 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import statistics 

accuracies = torch.load(os.path.join(os.getcwd(), 'params', 'accuracies_8T_5seeds.npy'))
random_seeds = np.arange(1000, 1020, 1)

plt.ylim(0.66, 0.70)
plt.plot(random_seeds, accuracies)
plt.title('Accuracies for different seeds for splitting the data', fontsize = 14)
plt.xlabel('random seed', fontsize = 14)
plt.ylabel('Accuracy on the test set', fontsize = 14)
plt.show()

accuracies = np.array(accuracies)

print(accuracies.mean())
print(accuracies.std())