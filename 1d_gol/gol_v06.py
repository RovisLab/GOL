import numpy as np
#import tensorflow as tf
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab

animate = False
anim_path = "C:/dev/src/ml/GOL/anim"


class TemplateObjects(object):
    def __init__(self):
        # Data features:    - position on x-axis
        #                   - amplitude on y-axis
        self.mu = np.array([-15, 6, 13])
        self.num_objects = self.mu.size


# Draws regularization samples from a template object with a certain eps
class RegularizationSamples(object):
    def __init__(self, template_object, num_reg_samples_per_object):
        self.mu = template_object.mu + 0.7
        self.sigma = 1.2
        self.samples = np.zeros((template_object.num_objects, num_reg_samples_per_object))
        for idx, mu in enumerate(self.mu):
            self.samples[idx] = np.random.normal(mu, self.sigma, num_reg_samples_per_object)


class GeneralizationGenerator(object):
    def __init__(self, range, num_objects, num_gen_samples_per_object):
        self.range = range
        self.num_gen_samples_per_object = num_gen_samples_per_object
        self.sigma = 1.5
        self.loss_threshold = 2.
        self.learning_rate = 0.3

        # Initialize the original generalization functions parameters
        self.mu = np.random.uniform(-range, range, num_objects)
        self.samples = np.zeros((num_objects, self.num_gen_samples_per_object))
        self.generate_samples()

    def generate_samples(self):
        for idx, mu in enumerate(self.mu):
            self.samples[idx] = np.random.normal(mu, self.sigma, self.num_gen_samples_per_object)

    # Recalculates the deformation parameters using the given losses between the regularization and
    # artificially generated samples
    def generalization_functions(self, losses, template_objects):
        # Regenerate artificial samples for each object type if the loss per object type is larger than the
        # loss threshold
        for idx, loss in enumerate(losses):
            if loss > self.loss_threshold:
                # Calculate the loss gradient
                err = (self.mu[idx] - template_objects.mu[idx])
                print(err)
                # Update the parameters
                self.mu[idx] = self.mu[idx] - err * self.learning_rate

        # Regenerate artificial samples based on the new generalization functions parameters
        self.generate_samples()


class GOL(object):
    def __init__(self, template_objects, regularization_samples, generalization_generator, classifier=None):
        self.template_objects = template_objects
        self.regularization_samples = regularization_samples
        self.generalization_generator = generalization_generator
        self.classifier = classifier

    def loss(self):
        rss = self.regularization_samples.samples
        gss = self.generalization_generator.samples
        ed = np.zeros(self.template_objects.num_objects)

        # Calculate the loss between each corresponding object the regularization and generated samples
        for idx, reg_sample in enumerate(rss):
            rs_size = len(reg_sample)
            gs_size = len(gss[idx])
            rs = np.reshape(reg_sample, (rs_size, 1))
            gs = np.reshape(gss[idx], (gs_size, 1))
            dist = cdist(rs, gs, metric='euclidean')
            ed[idx] = (1 / dist.size) * dist.sum()

        return ed

    def train(self):
        self.train_generator()
        #self.train_classifier()

    def train_generator(self, num_epochs=10):
        print("Training the Generalization Generator")

        iter_count = 0

        self._plot_distributions(iter_count)
        iter_count = iter_count + 1

        for i in range(num_epochs):
            # Calculate the loss between regularization and generated samples
            losses = self.loss()
            print("loss = ", losses)

            # Calculate the new deformation parameters using the obtained losses
            self.generalization_generator.generalization_functions(losses, self.template_objects)

            self._plot_distributions(iter_count)
            iter_count = iter_count + 1

    def train_classifier(self):
        print("Training the Neural Network Classifier")

        # Format the input labels
        self.classifier.train()

    def _samples(self, num_bins=300):
        bins = np.linspace(-self.generalization_generator.range, self.generalization_generator.range, num_bins)

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
        bin_size = x_axis_width / (self.generalization_generator.range * 2)
        amp = 0.3

        # Scale the regularization samples bins to the [0, 1] interval
        prs = prs / np.amax(prs)
        prs = prs / 3

        # x-axis
        p_x = np.linspace(-self.generalization_generator.range, self.generalization_generator.range, x_axis_width)

        f, ax = plt.subplots(figsize=(8, 3.5))

        # Draw the template objects at their position on the x axis
        for idx, mu in enumerate(to.mu):
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
            to_offset = ((self.generalization_generator.range + mu) * bin_size)
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
                     mlab.normpdf(p_x, self.generalization_generator.mu[idx] + draw_offset, self.generalization_generator.sigma),
                     color=color,
                     linewidth=1.6)

        # Draw the figure
        plt.xlim([-self.generalization_generator.range, self.generalization_generator.range])

        if animate:
            plt.ylim([0, 0.32])
            # File path
            file_path = anim_path + '/anim_fig_' + str(iter_count) + '.png'
            # file_path.append("/foo.png")
            print(file_path)
            plt.savefig(file_path)
        else:
            plt.ylim([0, 0.5])
            plt.title('1-D Generative One-Shot Learning')
            plt.xlabel('Data values')
            plt.ylabel('Probability density')
            plt.legend()

        plt.show()

'''
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
            self.NN = self.multilayer_perceptron(self.x, self.weights, self.biases)
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
    def multilayer_perceptron(self, x, weights, biases):
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

    def train(self, samples, labels):
        with tf.Session() as session:
            session.run(self.init)

            for epoch in range(self.training_epochs):
                avg_cost = 0.

                _, c = session.run([self.optimizer, self.loss], feed_dict={self.x: samples, self.y: labels})

                avg_cost += c

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print("Optimization Finished!")

    def test(self, samples, labels):
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.NN, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.x: samples, self.y: labels}))
'''

def main():
    to = TemplateObjects()

    model = GOL(
        TemplateObjects(),
        RegularizationSamples(to, num_reg_samples_per_object=10),
        GeneralizationGenerator(range=20, num_objects=to.num_objects, num_gen_samples_per_object=10000)
        # Classifier(num_classes=to.num_objects)
    )
    model.train()


if __name__ == '__main__':
    main()
