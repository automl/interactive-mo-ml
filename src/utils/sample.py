import ConfigSpace
from ConfigSpace import CategoricalHyperparameter


def grid_search(configspace, num_samples):
    return ConfigSpace.util.generate_grid(
        configspace,
        {
            k: num_samples
            for k, v in configspace.get_hyperparameters_dict().items()
            if type(v) != CategoricalHyperparameter
        },
    )


def random_search(configspace, num_samples):
    return configspace.sample_configuration(num_samples)
