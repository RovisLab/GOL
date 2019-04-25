import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import spline


num_objects = -1
accuracy = np.array([], dtype=float)
losses = np.array([], dtype=float)

with open('evaluation_v04.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        acc = float(row[1])
        accuracy = np.append(accuracy, acc)

        num_objects = int(row[2])

        loss = np.fromstring(row[3], dtype=float, count=num_objects, sep=',')
        losses = np.append(losses, loss)

    f.close()

losses = np.reshape(losses, (int(losses.shape[0] / num_objects), num_objects))
max_losses = np.amax(losses, axis=0)

# Normalize the losses
losses = losses / max_losses

T = np.arange(0, accuracy.shape[0], 1)
x = np.linspace(T.min(), T.max(), 300)

accuracy_smooth = spline(T, accuracy, x)
plt.plot(x, accuracy_smooth, label='Classification Accuracy', linewidth=1.5)

for i in range(losses.shape[1]):
    loss_smooth = spline(T, losses[:,i], x)
    obj_label = 'Generalization Energy Pattern ' + str(i+1)
    plt.plot(x, loss_smooth, label=obj_label)

plt.xlabel('Epochs')
plt.ylabel('Classification Accuracy vs. Generalization Energy')
#plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.legend(loc=4)
#plt.savefig("test.png")
plt.show()
