import numpy as np
import matplotlib.pyplot as plt

'''
Accuracy vs different NN architectures (bar plot)
'''
N = 6
fig, ax = plt.subplots(figsize=(7, 3.5))

LeNet_means = (98.79, 99.47, 96.61, 97.43, 98.48, 99.16)
AlexNet_means = (99.49, 99.93, 99.74, 99.89, 99.53, 99.28)
GoogleNet_means = (99.51, 99.95, 99.74, 99.90, 99.86, 99.45)
LeNet_std = (5, 7, 3, 7, 4, 9)
AlexNet_std = (8, 4, 6, 8, 5, 3)
GoogleNet_std = (5, 7, 3, 6, 2, 8)

ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

rects1 = ax.bar(ind, LeNet_means, width)
rects2 = ax.bar(ind + width, AlexNet_means, width)
rects3 = ax.bar(ind + width*2, GoogleNet_means, width)

# add some text for labels, title and axes ticks
#ax.set_xlabel('Object class')
ax.set_ylabel('Accuracy [%]')

ax.set_title('Accuracy of visual GOL with different $c(\hat{x})$ classifiers')
ax.set_xticks(ind + width)

ax.set_xticklabels(('Speed limits', 'Other prohibitions', 'Derestriction', 'Mandatory', 'Danger', 'Unique'))
plt.setp(ax.get_xticklabels(), fontsize=10, rotation=30)

ax.legend((rects1[0], rects2[0], rects3[0]),
          ('LeNet', 'AlexNet', 'GoogleNet'),
          loc='lower right')

plt.ylim([96, 100])

plt.tight_layout()
plt.show()
