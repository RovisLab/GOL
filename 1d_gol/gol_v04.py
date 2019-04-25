import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
import matplotlib.pyplot as plt


class TemplateObject(object):
    def __init__(self):
        # Data features:    - position on x-axis
        #                   - amplitude on y-axis
        self.num_classes = 2
        #self.mu = np.array([2, 5])
        self.mu = 2
        self.amp = 1


# Draws regularization samples from a template object with a certain eps
class RegularizationSamples(object):
    def __init__(self, template_object, num_reg_samples):
        self.mu = template_object.mu + 0.7
        self.sigma = 0.8
        self.samples = np.random.normal(self.mu, self.sigma, num_reg_samples)


class GeneralizationGenerator(object):
    def __init__(self, range, num_gen_samples):
        self.range = range
        self.num_gen_samples = num_gen_samples
        self.mu = -3
        self.sigma = 1.5
        self.samples = np.random.normal(self.mu, self.sigma, self.num_gen_samples)

    def generate(self, mu):
        self.mu = mu
        self.samples = np.random.normal(self.mu, self.sigma, self.num_gen_samples)


class GOL(object):
    def __init__(self, template_object, regularization_samples, generalization_generator):
        self.template_object = template_object
        self.regularization_samples = regularization_samples
        self.generalization_generator = generalization_generator

    def loss(self):
        rs = np.reshape(self.regularization_samples.samples, (len(self.regularization_samples.samples), 1))
        gs = np.reshape(self.generalization_generator.samples, (len(self.generalization_generator.samples), 1))

        dist = cdist(rs, gs, metric='euclidean')
        ed = (1 / dist.size) * dist.sum()

        return ed

    def train(self):
        self.train_generator()
        self.train_nn()

    def train_generator(self):
        print("Training the Generalization Generator")
        loss = self.loss()
        print("loss = ", loss)
        self._plot_distributions()

    def train_nn(self):
        print("Training the Neural Network")


    def _samples(self, num_bins=100):
        bins = np.linspace(-self.generalization_generator.range, self.generalization_generator.range, num_bins)

        # Regularization samples
        rs = self.regularization_samples.samples
        prs, _ = np.histogram(rs, bins=bins, density=True)

        # Generalization Generator
        gg = self.generalization_generator.samples
        pgg, _ = np.histogram(gg, bins=bins, density=True)

        return prs, pgg

    def _plot_distributions(self):
        prs, pgg = self._samples();

        # x-axis
        p_x = np.linspace(-self.generalization_generator.range, self.generalization_generator.range, len(prs))

        # Draw the template object
        to = np.zeros(np.shape(p_x))
        to[(self.generalization_generator.range + self.template_object.mu) *
           int(len(prs)/(self.generalization_generator.range*2))] = self.template_object.amp

        f, ax = plt.subplots(1)
        plt.bar(p_x, to, 0.2, color="red", label='template object')
        plt.bar(p_x, prs, 0.1, color="green", label='regularization samples')
        plt.plot(p_x, pgg, color="blue", label='generalization generator')
        plt.title('Generative One-Shot Learning')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()


def main():
    to=TemplateObject()

    model = GOL(
        to,
        RegularizationSamples(to, num_reg_samples=10),
        GeneralizationGenerator(range=8, num_gen_samples=10000)
    )
    model.train()


if __name__ == '__main__':
    main()
