# %%
import random

from csrank.dataset_reader import ChoiceDatasetGenerator, ObjectRankingDatasetGenerator
import csrank as cs
import numpy as np

random.seed(42)
np.random.seed(42)


def modify(original):
    return np.array(
        [
            np.array([np.array([object] * 10) for object in list(instance)])
            for instance in list(original)
        ]
    )


gen = ObjectRankingDatasetGenerator(n_instances=20, n_objects=3, n_features=2)
# gen = ChoiceDatasetGenerator(
#     dataset_type="pareto", n_instances=50, n_objects=10, n_features=2
# )
X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()


fate = cs.RankSVM()
fate.fit(X_train, Y_train)

print(fate.predict(X_test))

# %%
