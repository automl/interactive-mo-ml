import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import pcs_new

cs = CS.ConfigurationSpace()
cs.add_hyperparameter(CSH.CategoricalHyperparameter("a", choices=[1, 2, 3]))
# a, Type: Categorical, Choices: {1, 2, 3}, Default: 1

with open("configspace.pcs_new", "w") as fh:
    fh.write(pcs_new.write(cs))
# 27

with open("configspace.pcs_new", "r") as fh:
    deserialized_conf = pcs_new.read(fh)
print(deserialized_conf)
