import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import csv

animate = False
anim_path = "C:/dev/src/ml/GOL/anim"


class TemplateObjects(object):
    def __init__(self, num_test_samples=1000):
        # Data features: position on x-axis
        self.features = np.array([-12, 5, 14])
        self.num_objects = self.features.size
        self.test_samples = np.zeros((self.num_objects, num_test_samples))
        self.test_sigma = 3.

        for idx, mu in enumerate(self.features):
            self.test_samples[idx] = np.random.normal(mu, self.test_sigma, num_test_samples)


# Draws regularization samples from a template object with a certain eps
class RegularizationSamples(object):
    def __init__(self, template_objects, num_reg_samples_per_object):
        self.mu = template_objects.features + 0.3
        self.sigma = 1.
        self.samples = np.zeros((template_objects.num_objects, num_reg_samples_per_object))

        for idx, mu in enumerate(self.mu):
            self.samples[idx] = np.random.normal(mu, self.sigma, num_reg_samples_per_object)


class GeneralizationGenerator(object):
    def __init__(self, template_objects, regularization_samples, num_objects, num_gen_samples_per_object):
        self.template_objects = template_objects
        self.regularization_samples = regularization_samples
        self.num_gen_samples_per_object = num_gen_samples_per_object
        self.learning_rate = 0.5
        self.samples = np.zeros((num_objects, self.num_gen_samples_per_object))

        ''' Initialize the original generalization functions parameters '''
        # Init mean as with the value of the template objects
        self.mu = self.template_objects.features

        # Init the standard deviation with the standard deviation of the regularization samples
        self.sigmas = np.std(self.regularization_samples.samples, 1)

        # Generate initial samples
        self.generate_samples()

    def generate_samples(self):
        for idx, sigma in enumerate(self.sigmas):
            self.samples[idx] = np.random.normal(self.template_objects.features[idx],
                                                 sigma,
                                                 self.num_gen_samples_per_object)

    # Recalculates the deformation parameters using the generalization energy
    def generalization_functions(self, generalization_energy):
        # self.sigmas += 6.3
        self.sigmas[0] += 6.9
        self.sigmas[1] += 2.3
        self.sigmas[2] += 2.3

        # Regenerate artificial samples based on the new generalization functions parameters
        self.generate_samples()


class GOL:
    def __init__(self, template_objects, regularization_samples, generalization_generator, classifier=None, range=20):
#        super(GOL, self).__init__(1, 2)
#        self.types[:] = Real(0, 500)
        self.template_objects = template_objects
        self.regularization_samples = regularization_samples
        self.generalization_generator = generalization_generator
        self.classifier = classifier
        self.range = range
        self.energy = None

    def generalization_energy(self):
        rss = self.regularization_samples.samples
        gss = self.generalization_generator.samples
        ed = np.zeros(self.template_objects.num_objects)

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

        print("Generalization_energy = ", self.energy[0])
        print("Accuracy = ", self.classifier.accuracy)

        x = solution.variables[:]
        solution.objectives[:] = [self.energy[0], self.classifier.accuracy]

    def train_generator(self):
        print("Training the Generalization Generator")

        # Calculate the generalization energy between regularization and generated samples
        self.energy = self.generalization_energy()

        # Calculate the new deformation parameters using the obtained generalization_energy
        self.generalization_generator.generalization_functions(self.energy)

    def train_classifier(self):
        print("Training the Neural Network Classifier")

        # Get the training and test samples (features) and labels in a TF format
        training_features, training_labels = self._format_dataset(self.generalization_generator.samples)
        test_features, test_labels = self._format_dataset(self.template_objects.test_samples)

        # Train and test the classifier
        self.classifier.train(training_features, training_labels, test_features, test_labels)

    def _format_dataset(self, features):
        # Computer labels for the generated samples into a one-hot format
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
        bins = np.linspace(-self.range, self.range, num_bins)

        # Regularization samples
        rs = self.regularization_samples.samples
        prs = np.zeros((self.template_objects.num_objects, num_bins - 1))
        for idx, reg_sample in enumerate(rs):
            hist, _ = np.histogram(reg_sample, bins=bins, density=True)
            prs[idx] = hist

        # Generalization Generator
        gg = self.generalization_generator.samples
        pgg = np.zeros((self.template_objects.num_objects, num_bins - 1))
        for idx, gen_sample in enumerate(gg):
            hist, _ = np.histogram(gen_sample, bins=bins, density=True)
            pgg[idx] = hist

        return prs, pgg

    def _plot_distributions(self, iter_count):
        prs, pgg = self._samples()
        to = self.template_objects
        x_axis_width = len(prs[0])
        bin_size = x_axis_width / (self.range * 2)
        amp = 0.3

        # Scale the regularization samples bins to the [0, 1] interval
        prs = prs / np.amax(prs)
        prs = prs / 3

        # x-axis
        p_x = np.linspace(-self.range, self.range, x_axis_width)

        f, ax = plt.subplots(figsize=(8, 3.5))

        # Draw the template objects at their position on the x axis
        for idx, mu in enumerate(to.features):
            color = cm.hot(idx / 3 + 0.15)
            if idx == 1:
                color = 'red'
            elif idx == 2:
                color = 'blue'

            # Draw the regularization samples
            # label_regularization_samples = 'regularization samples ' + str(idx + 1)
            plt.bar(p_x, prs[idx], 0.5, color=color, alpha=0.4)

            # Draw the template objects at their position on the x axis
            to_x = np.zeros(np.shape(p_x))
            to_offset = ((self.range + mu) * bin_size)
            to_x[int(to_offset)] = amp
            # label_template_objects = 'template object ' + str(idx + 1)
            if not animate:
                label_template_objects = 'Object template ' + str(idx + 1)
                plt.bar(p_x, to_x, 1., color=color, label=label_template_objects, alpha=0.7)
            else:
                plt.bar(p_x, to_x, 1., color=color, alpha=0.7)

            # Draw the generalization generated samples
            # label_generalization_samples = 'generalization generator ' + str(idx + 1)
            # plt.plot(p_x, pgg[idx], color=color, linewidth=1.2)
            draw_offset = 0.5
            plt.plot(p_x,
                     mlab.normpdf(p_x,
                                  self.generalization_generator.mu[idx] + draw_offset,
                                  self.generalization_generator.sigmas[idx]),
                     color=color,
                     linewidth=1.6)

        # Draw the figure
        plt.xlim([-self.range, self.range])

        if animate:
            plt.ylim([0, 0.32])
            # File path
            file_path = anim_path + '/anim_fig_' + str(iter_count) + '.png'
            plt.savefig(file_path)
        else:
            plt.ylim([0, 0.6])
            # plt.title('1-D Generative One-Shot Learning')
            plt.xlabel('Data values')
            plt.ylabel('Probability density')
            plt.legend()

        plt.show()

    def fitness(self, x):
        #f1 = sum(x**2)
        #f2 = sum(x*x*x + 40*x)
        #print(type(f1))
        f1 = np.float64(np.random.rand(1))
        f2 = np.float64(np.random.rand(1))
        #print(f1, f2)
        return (f1, f2, )

    def get_bounds(self):
        return ([-1] * 2, [1] * 2)

    def get_nobj(self):
        return 2

    def get_name(self):
        return "Sphere Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(2)


class Classifier(object):
    def __init__(self, num_classes):
        # Network parameters
        self.num_classes = num_classes
        self.num_inputs = 1
        self.learning_rate = 0.003
        self.training_epochs = 1200
        self.display_step = 100
        self.num_hidden_1 = 10  # 1st layer number of features
        self.num_hidden_2 = 10  # 2nd layer number of features
        self.accuracy = None

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

        # Construct model and encapsulating all ops into scopes, making
        # Tensorboard's Graph visualization more convenient
        with tf.name_scope('Model'):
            # Model
            self.NN = self._multilayer_perceptron(self.x, self.weights, self.biases)
        with tf.name_scope('Loss'):
            # Softmax Cross entropy (cost function)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.NN, labels=self.y))
        with tf.name_scope('SGD'):
            # Gradient Descent
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        with tf.name_scope('Accuracy'):
            # Accuracy
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
        # Hidden layer with RELU activation
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
                #if epoch % self.display_step == 0:
                    #print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            #print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(self.NN, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            metric = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.accuracy = metric.eval({self.x: test_features, self.y: test_labels})
            # print("Accuracy:", self.accuracy)
            '''
            # Calculate the decision boundary
            num_points = features.shape[0]
            db = np.zeros((num_points, 1))
            for i in range(num_points // num_points):
                db[num_points * i:num_points * (i + 1)] = session.run(self.NN, {
                    self.x: np.reshape(
                        xs[num_points * i:num_points * (i + 1)],
                        (num_points, 1)
                    )
                })
            '''



def main():
    to = TemplateObjects()
    rs = RegularizationSamples(to, num_reg_samples_per_object=10)
    gg = GeneralizationGenerator(to, rs, num_objects=to.num_objects, num_gen_samples_per_object=100)
    cs = Classifier(num_classes=to.num_objects)

    model = GOL(to, rs, gg, cs)
    model.evaluate()


if __name__ == '__main__':
    main()
