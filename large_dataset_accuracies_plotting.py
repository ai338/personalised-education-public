import matplotlib.pyplot as plt 
import numpy as np 
# Accuracy on the test data in 2017 is 0.656 after training parameters; dataset size = (9247, 70)

# Accuracy on the test data in Nov2018 batch of tests is 0.681 after training parameters; dataset size = (12963, 88) = 1140744 datapoint pairs 

# Accuracy on the test data in Summer2019 is 0.666 after training parameters; dataset size = (17085, 58) = 990930 datapoint pairs 

n = 3 
error = 0.5
accuracies = [65.6, 66.6, 68.1]
data_size = [9247*70, 990930, 12963 * 88] 
# plt.scatter(data_size, accuracies)
# plt.show()

fig = plt.figure()
x = data_size
y = accuracies
yerr = np.repeat(0.5/np.sqrt(20), n)

coeffs = np.polyfit(x, y, 1)

plt.errorbar(x, y, yerr=yerr, label='acc. on test set', fmt='o', color = 'r')

xmin, xmax, ymin, ymax = plt.axis()
xx = np.arange(xmin, xmax, 10)
xx2 = np.square(xx)
yy = coeffs[1] + coeffs[0]*xx

plt.plot(xx, yy, linestyle='--', label = 'OLS linear fit')
plt.xlabel('number of points in the dataset', fontsize = 15)
plt.ylabel('accuracy on the test set', fontsize = 15)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14)
plt.title('Accuracy of the ordinal model as data size is increased', fontsize = 14)
plt.show()

