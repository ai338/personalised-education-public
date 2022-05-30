import matplotlib.pyplot as plt 
#  Accuracy on the test data in 2017 is 0.656 after training parameters; dataset size = (9247, 70)

# Accuracy on the test data in Nov2018 batch of tests is 0.681 after training parameters; dataset size = (12963, 88) = 1140744 datapoint pairs 

# Accuracy on the test data in Summer2019 is 0.666 after training parameters; dataset size = (17085, 58) = 990930 datapoint pairs 

accuracies = [65.6, 66.6, 68.1]
data_size = [9247*70, 990930, 12963 * 88] 

plt.scatter(data_size, accuracies)
plt.show()
