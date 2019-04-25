import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.interpolate import spline

class AnyObject(object):
    pass

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0,y0+width], [1.0*height,1.0*height], color='k')
        l2 = mlines.Line2D([x0,y0+width], [0.5*height,0.5*height], color='r')
        l3 = mlines.Line2D([x0, y0 + width], [0.0 * height, 0.0 * height], color='b')
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        return [l1, l2, l3]



num_objects = -1
accuracy = np.array([], dtype=float)
losses = np.array([], dtype=float)

with open('evaluation.csv', 'rt') as f:
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
losses = (losses / max_losses)

T = np.arange(0, accuracy.shape[0], 1)
x = np.linspace(T.min(), T.max(), 300)

accuracy_smooth = spline(T, accuracy, x)
line2, = plt.plot(x, accuracy_smooth, label='Accuracy', linewidth=1.5)

for i in range(losses.shape[1]):
    loss_smooth = spline(T, losses[:,i], x)
    plt.plot(x, loss_smooth)

plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.grid(True)

#plt.legend()
plt.legend(handles=[line2], loc=4)
plt.legend([AnyObject()], ['Generalization Energy'], handler_map={AnyObject: AnyObjectHandler()})


#plt.savefig("evaluation.png")
plt.show()
