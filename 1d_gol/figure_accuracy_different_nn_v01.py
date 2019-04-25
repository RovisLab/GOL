import numpy as np
import matplotlib.pyplot as plt

'''
Accuracy & generalization vs no of NN training iterations
'''
N = 3
fig, ax = plt.subplots()

Speed_limits_means = (82, 81, 83)
Other_prohibitions_means = (90, 31, 83)
Derestriction_means = (32, 63, 45)
Mandatory_means = (84, 25, 39)
Danger_means = (26, 35, 78)
Unique_means = (46, 75, 86)
Speed_limits_std = (3, 7, 2)
Other_prohibitions_std = (7, 8, 5)
Derestriction_std = (4, 2, 9)
Mandatory_std = (6, 3, 9)
Danger_std = (5, 8, 3)
Unique_std = (7, 3, 6)

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

rects1 = ax.bar(ind, Speed_limits_means, width, yerr=Speed_limits_std)
rects2 = ax.bar(ind + width, Other_prohibitions_means, width, yerr=Other_prohibitions_std)
rects3 = ax.bar(ind + width*2, Derestriction_means, width, yerr=Derestriction_std)
rects4 = ax.bar(ind + width*3, Mandatory_means, width, yerr=Mandatory_std)
rects5 = ax.bar(ind + width*4, Danger_means, width, yerr=Danger_std)
rects6 = ax.bar(ind + width*5, Unique_means, width, yerr=Unique_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy results for visual GOL equipped with different neural network classifiers')
ax.set_xticks(ind + width / 2)

ax.set_xticklabels(('LeNet', 'AlexNet', 'GoogleNet'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),
          ('Speed limits', 'Other prohibitions', 'Derestriction', 'Mandatory', 'Danger', 'Unique'))

plt.show()

'''
Accuracy & generalization vs different NN architectures (bar plot)
'''

'''
Accuracy & generalization vs different no of neurons
'''

'''
Accuracy & generalization vs no of generated samples
'''
