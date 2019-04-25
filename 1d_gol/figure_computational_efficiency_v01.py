import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

'''
Accuracy vs no of NN training iterations
Accuracy vs different no of neurons
Accuracy vs no of generated samples
'''


def normalize(x):
    return (x-min(x))/(max(x)-min(x))


no_synthetic_samples = np.arange(0, 20, 1)
training_time_lenet = np.zeros((20)) #np.array([0, 0.31, 0.89, 0.91, 0.97, 0.99, 0.99, 0.98, 0.99, 0.97, 0.99, 0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.99])
training_time_alexnet = np.zeros((20)) #np.array([0, 0.31, 0.89, 0.91, 0.97, 0.99, 0.99, 0.98, 0.99, 0.97, 0.99, 0.99, 0.98, 0.99, 0.3, 0.98, 0.99, 0.99, 0.99, 0.99])
training_time_googlenet = np.zeros((20)) #np.array([0, 0.31, 0.89, 0.91, 0.97, 0.99, 0.5, 0.98, 0.99, 0.97, 0.99, 0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.99])

f, ax = plt.subplots(figsize=(5, 2.5))
plt.title('GOL computational efficiency', y=1.0)

plt.grid(linestyle='--')
plt.xlabel('No of generated synthetic samples')
plt.ylabel('Time')
plt.xticks(np.arange(20), ('1', '', '', '300', '', '', '600', '', '', '900',
                           '', '', '1200', '', '', '1500', '', '', '1800', ''))
plt.plot(no_synthetic_samples, training_time_lenet, label='LeNet', linewidth=1.5)
plt.plot(no_synthetic_samples, training_time_alexnet, label='AlexNet', linewidth=1.5)
plt.plot(no_synthetic_samples, training_time_googlenet, label='GoogleNet', linewidth=1.5)

plt.tight_layout()
plt.xlim([0, 20])
plt.ylim([0, 100])
plt.legend(loc='lower right', fancybox=False, shadow=False)
plt.show()
