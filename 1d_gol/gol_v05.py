import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class TemplateObjects(object):
    def __init__(self):
        # Data features:    - position on x-axis
        #                   - amplitude on y-axis
        self.mu = np.array([-15, 6, 13])
        self.num_objects = self.mu.size


# Draws regularization samples from a template object with a certain eps
class RegularizationSamples(object):
    def __init__(self, template_object, num_reg_samples_per_object):
        self.mu = template_object.mu + 0.
        self.sigma = 0.8
        self.samples = np.zeros((template_object.num_objects, num_reg_samples_per_object))
        for idx, mu in enumerate(self.mu):
            self.samples[idx] = np.random.normal(mu, self.sigma, num_reg_samples_per_object)


class GeneralizationGenerator(object):
    def __init__(self, range, num_objects, num_gen_samples_per_object):
        self.range = range
        self.num_gen_samples_per_object = num_gen_samples_per_object
        self.sigma = 1.5
        self.loss_threshold = 2.
        self.learning_rate = 1.

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
    def __init__(self, template_objects, regularization_samples, generalization_generator):
        self.template_objects = template_objects
        self.regularization_samples = regularization_samples
        self.generalization_generator = generalization_generator

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
        #self.train_nn()

    def train_generator(self, num_epochs=10):
        print("Training the Generalization Generator")

        for i in range(num_epochs):
            # Calculate the loss between regularization and generated samples
            losses = self.loss()
            print("loss = ", losses)

            # Calculate the new deformation parameters using the obtained losses
            self.generalization_generator.generalization_functions(losses, self.template_objects)

        self._plot_distributions()

    def train_nn(self):
        print("Training the Neural Network")

    def _samples(self, num_bins=100):
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

    def _plot_distributions(self):
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

        f, ax = plt.subplots(1)

        # Draw the template objects at their position on the x axis
        for idx, mu in enumerate(to.mu):
            color = cm.hot(idx / 3 + 0.15)

            # Draw the regularization samples
            # label_regularization_samples = 'regularization samples ' + str(idx + 1)
            plt.bar(p_x, prs[idx], 0.5, color=color, alpha=0.6)

            # Draw the template objects at their position on the x axis
            to_x = np.zeros(np.shape(p_x))
            to_offset = ((self.generalization_generator.range + mu) * bin_size)
            to_x[int(to_offset)] = amp
            #label_template_objects = 'template object ' + str(idx + 1)
            label_template_objects = 'Object type ' + str(idx + 1)
            plt.bar(p_x, to_x, 1., color=color, label=label_template_objects)

            # Draw the generalization generated samples
            #label_generalization_samples = 'generalization generator ' + str(idx + 1)
            plt.plot(p_x, pgg[idx], color=color, linewidth=1.2)

        # Draw the figure
        plt.xlim([-self.generalization_generator.range, self.generalization_generator.range])
        plt.ylim([0, 0.5])
        plt.title('Generative One-Shot Learning')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()


def main():
    to = TemplateObjects()

    model = GOL(
        to,
        RegularizationSamples(to, num_reg_samples_per_object=10),
        GeneralizationGenerator(range=20, num_objects=to.num_objects, num_gen_samples_per_object=10000)
    )
    model.train()


if __name__ == '__main__':
    main()
