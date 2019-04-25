import numpy as np
import matplotlib.pyplot as plt


class TemplateObject(object):
    def __init__(self):
        self.mu = 2


class RegularizationSamples(object):
    def __init__(self):
        self.mu = 2.5
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneralizationGenerator(object):
    def __init__(self, range):
        self.range = range
        self.mu = -3
        self.sigma = 1.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GOL(object):
    def __init__(self, template_object, regularization_samples, generalization_generator):
        self.template_object = template_object
        self.regularization_samples = regularization_samples
        self.generalization_generator = generalization_generator

    def train(self):
        self._plot_distributions()

    def _samples(self, num_points=10000, num_regularization_samples=10, num_bins=100):
        xs = np.linspace(-self.generalization_generator.range, self.generalization_generator.range, num_points)
        bins = np.linspace(-self.generalization_generator.range, self.generalization_generator.range, num_bins)

        # Regularization samples
        rs = self.regularization_samples.sample(num_regularization_samples)
        prs, _ = np.histogram(rs, bins=bins, density=True)

        # Generalization Generator
        gg = self.generalization_generator.sample(num_points)
        pgg, _ = np.histogram(gg, bins=bins, density=True)

        return prs, pgg

    def _plot_distributions(self):
        prs, pgg = self._samples();
        p_x = np.linspace(-self.generalization_generator.range, self.generalization_generator.range, len(prs))
        to = np.zeros(np.shape(p_x))
        to[(self.generalization_generator.range + self.template_object.mu) * int(len(prs)/(self.generalization_generator.range*2))] = 1
        f, ax = plt.subplots(1)
        plt.bar(p_x, to, 0.2, color="red", label='template object')
        plt.plot(p_x, prs, label='regularization samples')
        plt.plot(p_x, pgg, label='generalization generator')
        plt.title('Generative One-Shot Learning')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()


def main():
    model = GOL(
        TemplateObject(),
        RegularizationSamples(),
        GeneralizationGenerator(range=8)
    )
    model.train()


if __name__ == '__main__':
    main()
