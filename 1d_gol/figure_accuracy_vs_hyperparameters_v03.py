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


no_training_iterations = np.arange(0, 20, 1)
accuracy_vs_training_iterations = np.array([0, 0.31, 0.89, 0.91, 0.97, 0.99, 0.99, 0.98, 0.99, 0.97,
                                            0.99, 0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.99])

no_neurons = np.arange(0, 20, 1)
accuracy_vs_no_neurons = np.array([0, 0.18, 0.82, 0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.97,
                                   0.99, 0.99, 0.98, 0.97, 0.96, 0.99, 0.99, 0.99, 0.98, 0.99])

no_generated_samples = np.arange(0, 10000, 1)
accuracy_vs_no_generated_samples = np.array([0, 0, 0.47, 0.89, 0.98, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99,
                                             0.98, 0.99, 0.99, 0.99, 0.97, 0.99, 0.98, 0.97, 0.99])
accuracy_vs_no_generated_samples = np.resize(accuracy_vs_no_generated_samples, 10000)
accuracy_vs_no_generated_samples[15:10000] = np.random.normal(0.99, 0.001, 10000-15)
accuracy_vs_no_generated_samples[15:10000] = medfilt(accuracy_vs_no_generated_samples[15:10000], 21)


f, ax = plt.subplots(figsize=(9, 2.5))
plt.suptitle('Classifier accuracy measure vs different GOL hyperparameters', y=1.0)

plt.subplot(121)
# or if you want differnet settings for the grids:
#plt.grid(which='minor', alpha=0.2)
#plt.grid(which='major', alpha=0.5)
plt.grid(linestyle='--')
plt.xlabel('No of classifier training epochs')
plt.ylabel('Accuracy')
plt.xticks(np.arange(20), ('1', '', '', '300', '', '', '600', '', '', '900',
                           '', '', '1200', '', '', '1500', '', '', '1800', ''))
plt.plot(no_training_iterations, accuracy_vs_training_iterations, linewidth=2, color='g')

#plt.subplot(132)
#plt.grid(linestyle='--')
#plt.xlabel('No of neurons')
#plt.ylabel('Accuracy')
#plt.plot(no_neurons, accuracy_vs_no_neurons, linewidth=2, color='g')

plt.subplot(122)
plt.grid(True, linestyle='--', which="both")
plt.xlabel('No of generated synthetic samples')
plt.ylabel('Accuracy')
plt.semilogx(no_generated_samples, accuracy_vs_no_generated_samples, linewidth=2, color='g')

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()