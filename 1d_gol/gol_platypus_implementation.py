import numpy as np
#np.set_printoptions(threshold=np.nan)
from platypus import NSGAII, Problem, Real
import tensorflow as tf
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from scipy.signal import medfilt
import csv

animate = True
anim_path = "C:/dev/src/ml/GOL/anim"


class OneShotObjects(object):
    def __init__(self, num_test_samples_per_object=1000, range=20):
        # Data features: position on x-axis
        self.features = np.array([-12, 5, 14])
        self.num_objects = self.features.size
        self.test_samples = np.zeros((self.num_objects, num_test_samples_per_object))
        self.test_sigma = 2.0
        self.range = range

        for idx, mu in enumerate(self.features):
            self.test_samples[idx] = np.random.normal(mu, self.test_sigma, num_test_samples_per_object)


# Draws regularization samples from a one-shot object with a certain eps
class RegularizationSamples(object):
    def __init__(self, one_shot_objects, num_reg_samples_per_object):
        self.mu = one_shot_objects.features + 0.3
        self.sigma = 1.
        self.samples = np.zeros((one_shot_objects.num_objects, num_reg_samples_per_object))

        for idx, mu in enumerate(self.mu):
            self.samples[idx] = np.random.normal(mu, self.sigma, num_reg_samples_per_object)


class GeneralizationGenerator(object):
    def __init__(self, one_shot_objects, regularization_samples, num_objects, num_gen_samples_per_object):
        self.one_shot_objects = one_shot_objects
        self.regularization_samples = regularization_samples
        self.num_gen_samples_per_object = num_gen_samples_per_object
        self.generalization_rate = 0.1
        self.samples = np.zeros((num_objects, self.num_gen_samples_per_object))
        self.mu = np.empty(shape=np.shape(self.one_shot_objects.features))

        ''' Initialize the generalization functions parameters '''
        # self.mu = self.one_shot_objects.features      # Init mean as with the value of the one-shot objects
        self.mu = np.random.uniform(-self.one_shot_objects.range,
                                    self.one_shot_objects.range,
                                    np.shape(self.one_shot_objects.features))

        # Init the standard deviation with the standard deviation of the regularization samples
        self.sigmas = np.std(self.regularization_samples.samples, 1)

        # Generate initial samples
        self.generate_samples()

    def generate_samples(self):
        for idx, sigma in enumerate(self.sigmas):
            self.samples[idx] = np.random.normal(self.one_shot_objects.features[idx],
                                                 sigma,
                                                 self.num_gen_samples_per_object)

    # Recalculates the deformation parameters using the generalization energy
    def generalization_functions(self, generalization_energy):
        # Center the mean value on the one-shot objects
        self.mu = self.one_shot_objects.features

        # Update the learned generation parameter sigma
        self.sigmas += self.generalization_rate

        # Regenerate artificial samples based on the new generalization functions parameters
        self.generate_samples()


class GOL(Problem):
    def __init__(self, one_shot_objects, regularization_samples, generalization_generator, classifier=None):
        super(GOL, self).__init__(1, 2)
        self.types[:] = Real(0, 500)
        self.one_shot_objects = one_shot_objects
        self.regularization_samples = regularization_samples
        self.generalization_generator = generalization_generator
        self.classifier = classifier
        self.gen_energy = []
        self.objectives = []
        self.epoch_count = 0

        # Plot the initial distribution of the GOL model parameters
        self._plot_distributions(self.epoch_count)

    def generalization_energy(self):
        rss = self.regularization_samples.samples
        gss = self.generalization_generator.samples
        ed = np.zeros(self.one_shot_objects.num_objects)

        # Calculate the generalization energy between each corresponding object in the regularization samples and
        # the generated samples
        for idx, reg_sample in enumerate(rss):
            rs_size = len(reg_sample)
            gs_size = len(gss[idx])
            rs = np.reshape(reg_sample, (rs_size, 1))
            gs = np.reshape(gss[idx], (gs_size, 1))
            dist = cdist(rs, gs, metric='euclidean')
            ed[idx] = (1 / dist.size) * dist.sum()

        return ed

    def evaluate(self, solution):
        self.train_generator()
        self.train_classifier()

        mean_energy = np.mean(self.gen_energy)

        print('++ epoch', '%03d' % self.epoch_count, "generalization energy", "{:.4f}".format(mean_energy))
        print('++ epoch', '%03d' % self.epoch_count, "classification accuracy", "{:.4f}".format(self.classifier.accuracy))

        x = solution.variables[:]
        solution.objectives[:] = [mean_energy, self.classifier.accuracy]

        num_energies = np.shape(self.gen_energy)[0]
        t = np.reshape(self.gen_energy, (num_energies, 1))
        t = np.append(t, self.classifier.accuracy)
        t = np.reshape(t, (num_energies+1, 1))
        self.objectives.append(t)

        self.epoch_count += 1
        self._plot_distributions(self.epoch_count)

    def train_generator(self):
        print("--- Generalization Generator training epoch ", self.epoch_count)

        # Calculate the generalization energy between regularization and generated samples
        self.gen_energy = self.generalization_energy()

        # Calculate the new deformation parameters using the obtained generalization_energy
        self.generalization_generator.generalization_functions(self.gen_energy)

    def train_classifier(self):
        print("--- Classifier training epoch ", self.epoch_count)

        # Get the training and test samples (features) and labels in a TF format
        training_features, training_labels = self._format_dataset(self.generalization_generator.samples)
        test_features, test_labels = self._format_dataset(self.one_shot_objects.test_samples)

        # Train and test the classifier
        self.classifier.train(training_features, training_labels, test_features, test_labels)

    def _format_dataset(self, features):
        # Compute labels for the generated samples into a one-hot format
        num_classes = features.shape[0]
        num_samples = features.shape[1]

        batch_labels = np.empty(shape=(0, 0))

        for idx in range(num_classes):
            labels = np.zeros((num_samples, num_classes), float)
            labels[:num_samples, idx:idx + 1] = 1.
            batch_labels = np.append(batch_labels, labels)

        batch_labels = np.reshape(batch_labels, (num_classes * num_samples, num_classes))

        # Reshape the features
        batch_features = np.reshape(features, (num_classes * num_samples, 1))

        return batch_features, batch_labels

    def _samples(self, num_bins=300):
        bins = np.linspace(-self.one_shot_objects.range, self.one_shot_objects.range, num_bins)

        # Regularization samples
        rs = self.regularization_samples.samples
        prs = np.zeros((self.one_shot_objects.num_objects, num_bins - 1))
        for idx, reg_sample in enumerate(rs):
            hist, _ = np.histogram(reg_sample, bins=bins, density=True)
            prs[idx] = hist

        # Generalization Generator
        gg = self.generalization_generator.samples
        pgg = np.zeros((self.one_shot_objects.num_objects, num_bins - 1))
        for idx, gen_sample in enumerate(gg):
            hist, _ = np.histogram(gen_sample, bins=bins, density=True)
            pgg[idx] = hist

        return prs, pgg

    def _plot_distributions(self, epoch_count):
        prs, pgg = self._samples()
        to = self.one_shot_objects
        x_axis_width = len(prs[0])
        bin_size = x_axis_width / (self.one_shot_objects.range * 2)
        amp = 0.3

        #fig_data = plt.figure(1)
        f, ax = plt.subplots(figsize=(8, 6))

        # Scale the regularization samples bins to the [0, 1] interval
        prs = prs / np.amax(prs)
        prs = prs / 3

        # x-axis
        p_x = np.linspace(-self.one_shot_objects.range, self.one_shot_objects.range, x_axis_width)

        '''
        Draw the one-shot data, generated distributions and decision boundaries
        '''
        plt.subplot(211)

        # Draw the one-shot objects at their position on the x axis
        for idx, mu in enumerate(to.features):
            color = cm.hot(idx / 3 + 0.15)
            if idx == 1:
                color = 'red'
            elif idx == 2:
                color = 'blue'

            # Draw the regularization samples
            plt.bar(p_x, prs[idx], 0.5, color=color, alpha=0.4)

            # Draw the one-shot objects at their position on the x axis
            to_x = np.zeros(np.shape(p_x))
            to_offset = ((self.one_shot_objects.range + mu) * bin_size)
            to_x[int(to_offset)] = amp

            label_one_shot_objects = 'One-shot object ' + str(idx + 1)
            plt.bar(p_x, to_x, 1., color=color, label=label_one_shot_objects, alpha=0.7)

            # Draw the generalization generated samples
            draw_offset = 0.5

            sigma = self.generalization_generator.sigmas[idx]
            generated_distribution = mlab.normpdf(p_x, self.generalization_generator.mu[idx] + draw_offset, sigma)

            max_idx = np.argmax(generated_distribution)
            indices = np.arange(0, np.shape(generated_distribution)[0], 1)
            scaling_offset = np.absolute(indices - max_idx)
            scaling_offset = scaling_offset / np.max(scaling_offset)
            scaling_offset = np.reshape(scaling_offset, np.shape(generated_distribution))

            dist_low_bound = (generated_distribution + sigma * 0.03) - scaling_offset * 0.2
            dist_up_bound = (generated_distribution - sigma * 0.03) - scaling_offset * 0.2

            plt.plot(p_x,
                     generated_distribution,
                     color=color,
                     linewidth=1.6)
            plt.fill_between(p_x,
                             dist_low_bound,
                             dist_up_bound,
                             color=color,
                             alpha=0.2)

            # Draw the decision boundaries
            db = self.classifier.decision_boundaries[:, idx]
            db_smooth = medfilt(db, 99)
            db_smooth -= 0.5
            db_x = np.linspace(-self.one_shot_objects.range, self.one_shot_objects.range, len(db_smooth))
            plt.plot(db_x, db_smooth, label='decision boundary ' + str(idx + 1), linewidth=1.0, linestyle='--')

        # Draw the figure
        plt.xlim([-self.one_shot_objects.range, self.one_shot_objects.range])
        plt.ylim([0, 0.7])
        plt.title('1-D Generative One-Shot Learning - Training Episode ' + str(epoch_count))
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=False, shadow=False)

        '''
        Draw the generalization energies and classification accuracy
        '''
        plt.subplot(212)
        plt.xlim([0, 100])
        plt.ylim([0, 1.37])
        plt.xlabel('Training episode')
        plt.ylabel('Objectives')

        if epoch_count != 0:
            num_objectives = np.shape(self.objectives)[1]

            for idx in range(num_objectives):
                x = np.arange(0, 100, 1)
                y = np.zeros(np.shape(x))
                y[:] = np.NAN

                t = np.array(self.objectives)[:, idx]

                # Normalize the generalization energies for visualization
                if idx != num_objectives - 1:
                    y[:len(t[:, 0])] = (t[:, 0] - 0.8) / 8.2
                else:
                    y[:len(t[:, 0])] = t[:, 0]

                y_gradient = np.gradient(y)

                if idx != num_objectives - 1:
                    label_objective = 'Generalization energy ' + str(idx + 1)
                else:
                    label_objective = 'Classification accuracy'

                plt.plot(x, y, label=label_objective)
                plt.fill_between(x, y-y_gradient, y+y_gradient, alpha='0.5')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fancybox=False, shadow=False)

        if animate:
            # File path
            file_path = anim_path + '/anim_fig_' + '%05d' % epoch_count + '.png'
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()


class Classifier(object):
    def __init__(self, num_classes, range, learning_rate, training_epochs, display_step, num_hidden_1, num_hidden_2):
        # Network parameters
        self.num_classes = num_classes
        self.range = range
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.display_step = display_step
        self.num_hidden_1 = num_hidden_1  # 1st layer number of features
        self.num_hidden_2 = num_hidden_2  # 2nd layer number of features
        self.num_inputs = 1
        self.accuracy = None
        self.decision_boundaries = np.zeros((0, self.num_classes))
        self.db_num_points = 3000

        # tf Graph Input
        # input data image of shape 1 (one feature)
        self.x = tf.placeholder(tf.float32, [None, self.num_inputs], name='InputData')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='LabelData')

        # Store layers weight & bias
        self.weights = {
            'w1': tf.Variable(tf.random_normal([self.num_inputs, self.num_hidden_1]), name='W1'),
            'w2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2]), name='W2'),
            'w3': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_classes]), name='W3')
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.num_hidden_1]), name='b1'),
            'b2': tf.Variable(tf.random_normal([self.num_hidden_2]), name='b2'),
            'b3': tf.Variable(tf.random_normal([self.num_classes]), name='b3')
        }

        # Construct NN classifier model
        with tf.name_scope('Model'):
            # Multilayer Perceptron Model
            self.NN = self._multilayer_perceptron(self.x, self.weights, self.biases)

        with tf.name_scope('Loss'):
            # Softmax Cross entropy (cost function)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.NN, labels=self.y))

        with tf.name_scope('SGD'):
            # Gradient Descent
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('Accuracy'):
            # Accuracy measure
            self.acc = tf.equal(tf.argmax(self.NN, 1), tf.argmax(self.y, 1))
            self.acc = tf.reduce_mean(tf.cast(self.acc, tf.float32))

        # Initializing the variables
        self.init = tf.global_variables_initializer()

    # Neural Network model
    def _multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        # Create a summary to visualize the first layer ReLU activation
        tf.summary.histogram("relu1", layer_1)

        # Hidden layer with ReLU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

        # Create another summary to visualize the second layer ReLU activation
        tf.summary.histogram("relu2", layer_2)

        # Output layer
        out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

        return out_layer

    def train(self, train_features, train_labels, test_features, test_labels):
        with tf.Session() as session:
            session.run(self.init)

            for epoch in range(self.training_epochs):
                avg_cost = 0.

                _, c = session.run([self.optimizer, self.loss], feed_dict={self.x: train_features, self.y: train_labels})

                avg_cost += c

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Classifier training epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print("+++ Classifier training finished")

            # Calculate the decision boundaries
            db = np.linspace(-self.range, self.range, self.db_num_points)
            db = np.reshape(db, (self.db_num_points, 1))
            self.decision_boundaries = session.run(tf.nn.softmax(self.NN), feed_dict={
                self.x: db
            })

            # Test model
            correct_prediction = tf.equal(tf.argmax(self.NN, 1), tf.argmax(self.y, 1))

            # Calculate accuracy
            metric = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.accuracy = metric.eval({self.x: test_features, self.y: test_labels})


def main():
    # One-shot objects parameters
    init_num_test_samples_per_object = 1000
    init_range = 20

    # Regularization samples parameters
    init_num_reg_samples_per_object = 10

    # Generalization generator parameters
    # TODO: num_objects = os.num_objects,
    init_num_gen_samples_per_object = 300

    # Classifier parameters
    # TODO: self.num_classes = num_classes (num_objects)
    init_learning_rate = 0.003
    init_training_epochs = 300  # 1200
    init_display_step = 100
    init_num_hidden_1 = 10  # 1st layer number of features
    init_num_hidden_2 = 10  # 2nd layer number of features

    os = OneShotObjects(num_test_samples_per_object=init_num_test_samples_per_object,
                        range=init_range)
    rs = RegularizationSamples(one_shot_objects=os,
                               num_reg_samples_per_object=init_num_reg_samples_per_object)
    gg = GeneralizationGenerator(one_shot_objects=os,
                                 regularization_samples=rs,
                                 num_objects=os.num_objects,
                                 num_gen_samples_per_object=init_num_gen_samples_per_object)
    cs = Classifier(num_classes=os.num_objects,
                    range=init_range,
                    learning_rate=init_learning_rate,
                    training_epochs=init_training_epochs,
                    display_step=init_display_step,
                    num_hidden_1=init_num_hidden_1,
                    num_hidden_2=init_num_hidden_2)

    gol_model = GOL(os, rs, gg, cs)

    opt_algorithm = NSGAII(gol_model)
    opt_algorithm.run(2)

    plt.figure(2)
    plt.scatter([s.objectives[0] for s in opt_algorithm.result],
                [s.objectives[1] for s in opt_algorithm.result])
    plt.xlabel("$J(\Theta, \hat{x}(t))$")
    plt.ylabel("$a(\Theta, \hat{x}(t))$")

    if animate:
        # Save pareto solutions image
        file_path = anim_path + '/_pareto_solutions.png'
        plt.savefig(file_path)
        plt.close()

        # Save the pareto solutions
        file_path = anim_path + '/_pareto_solutions.csv'

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for s in opt_algorithm.result:
                writer.writerow((s.objectives[0], s.objectives[1]))
            f.close()
    else:
        plt.show()


if __name__ == '__main__':
    main()
