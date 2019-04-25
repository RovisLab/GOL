import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

'''
Accuracy & generalization vs no of NN training iterations
Accuracy & generalization vs different no of neurons
Accuracy & generalization vs no of generated samples
'''


def normalize(x):
    return (x-min(x))/(max(x)-min(x))


no_training_iterations = np.arange(0, 20, 1)
accuracy = np.array([0.34, 0.45, 0.89, 0.91, 0.97, 0.99, 0.99, 0.98, 0.99, 0.97,
                     0.99, 0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.99])
generalization_1 = np.array([7.06340598744, 7.12295012337, 7.35673947026, 7.39163962966, 7.3554818596, 7.56299908735,
                    7.6052485944, 7.52304338236, 7.99166883586, 7.88550581282, 7.8819218624, 7.99213870223,
                    8.19881628984, 8.02163077612, 8.46921175157, 8.43239304994, 8.57018012397, 8.24208212,
                    8.56927855174, 8.85112679007])
generalization_2 = np.array([6.94443781048, 6.9737130777, 7.17319539005, 7.21528533444, 7.2440367028, 7.21040208822,
                    7.5942428942, 7.64134750392, 7.5259996293, 7.8154758902, 7.75926875083, 7.88422672974,
                    8.06425963483, 7.90829482207, 8.27028621938, 8.21108932401, 8.38650541425, 8.34814778807,
                    8.332179531, 8.59904624691])
generalization_3 = np.array([7.16870251088, 7.20753204075, 7.28142325829, 7.37389043498, 7.36436098471, 7.5281938499,
                    7.51471036608, 7.52366923458, 7.79847616463, 7.65295408219, 7.86908956295, 7.82591277935,
                    8.04634330055, 8.17008616906, 8.19112211182, 8.13611638555, 8.56966391294, 8.67268108078,
                    8.43423460825, 8.63531908752])
generalization_4 = np.array([7.10263709646, 6.91062672426, 7.37529737907, 7.53496279784, 7.03409213994, 7.43542386056,
                    7.602459696, 8.48407015566, 7.89284393675, 8.09342005038, 8.14866069721, 8.06854738116,
                    7.76488617262, 7.97786927781, 8.65638492717, 8.51384304197, 8.73722641683, 8.30505553223,
                    8.70932482076, 9.00018372867])
generalization_5 = np.array([7.4803661669, 7.67097724215, 6.5325521123, 7.48649415738, 7.5349707647, 7.31203188901,
                    7.29054065564, 8.46170746646, 7.71976924089, 8.05556866027, 7.37951778752, 8.24751709854,
                    8.05597977051, 8.46981688664, 8.71299211747, 7.96891253492, 8.43208236479, 9.19975163014,
                    9.64491871238, 9.85144219952])
generalization_6 = np.array([6.47348438237, 7.50680395465, 7.49785314154, 6.84803070104, 7.16901886219, 7.38348372027,
                    7.08094064457, 7.60151232304, 7.13599955582, 7.79560394105, 7.56029250972, 8.6317972425,
                    8.88736254077, 8.02793372201, 8.29918617032, 8.4135189088, 9.84763932295, 9.31421964174,
                    9.06742035477, 9.72805717592])
generalization_7 = np.array([5.56017027217, 5.70204474626, 5.7985597515, 5.74817341122, 5.95773624285, 6.13520676457,
                    6.11074662477, 6.22765174111, 6.21556780137, 6.25351378016, 6.26613520635, 6.52451565874,
                    6.52197660679, 6.65572986379, 6.69069453374, 6.81955648612, 6.80692587427, 6.76331869754,
                    7.01740398807, 7.21367522355])
generalization_8 = np.array([5.34607556544, 5.39591297695, 5.69663316774, 5.57854205805, 5.70699004014, 5.79042316662,
                    5.92217939822, 5.98764504499, 6.0432698587, 6.12946881615, 6.29697851262, 6.33298102958,
                    6.50905385233, 6.52320256278, 6.66571918042, 6.62546128036, 6.65894919129, 6.8162025485,
                    6.82158711231, 7.00931030852])
generalization_9 = np.array([5.53719363212, 5.65422499782, 5.62727199775, 5.66417981545, 5.8505820292, 5.92488311196,
                    5.98155684531, 6.0136282754, 6.13276078106, 6.06949845702, 6.27178490012, 6.17940138828,
                    6.4729729529, 6.51232673399, 6.59845220399, 6.60877727207, 6.80930842583, 6.92071472362,
                    7.03987667568, 6.72094075778])
generalization_10 = np.array([5.72716426308, 5.70559538667, 6.32948577817, 6.02155500957, 6.13261283788, 6.5015268583,
                     5.85836197458, 6.39513459302, 6.13876736657, 6.58767042558, 6.21101819844, 6.34088774913,
                     6.4734326609, 7.06811089315, 7.18017911131, 6.12359523702, 7.2248528892, 6.6717310772,
                     7.26114233179, 7.14189068886])

filter_size = 7
generalization_1_normalized = medfilt(normalize(generalization_1), filter_size)
generalization_2_normalized = medfilt(normalize(generalization_2), filter_size)
generalization_3_normalized = medfilt(normalize(generalization_3), filter_size)
generalization_4_normalized = medfilt(normalize(generalization_4), filter_size)
generalization_5_normalized = medfilt(normalize(generalization_5), filter_size)
generalization_6_normalized = medfilt(normalize(generalization_6), filter_size)
generalization_7_normalized = medfilt(normalize(generalization_7), filter_size)
generalization_8_normalized = medfilt(normalize(generalization_8), filter_size)
generalization_9_normalized = medfilt(normalize(generalization_9), filter_size)
generalization_10_normalized = medfilt(normalize(generalization_10), filter_size)


f, ax = plt.subplots(figsize=(8, 6))

#plt.subplot(231)
# or if you want differnet settings for the grids:
#plt.grid(which='minor', alpha=0.2)
#plt.grid(which='major', alpha=0.5)
plt.grid(linestyle='--')

plt.xlabel('No of classifier training epochs')
plt.ylabel('Accuracy')
plt.plot(no_training_iterations, accuracy, linewidth=2, color='g')
plt.plot(no_training_iterations, generalization_1_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_2_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_3_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_4_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_5_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_6_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_7_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_8_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_9_normalized, linewidth=0.8)
plt.plot(no_training_iterations, generalization_10_normalized, linewidth=0.8)
plt.show()