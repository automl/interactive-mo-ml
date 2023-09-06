import numpy as np

from pymoo.indicators.hv import Hypervolume
from performance.r2 import R2
from performance.spacing import Spacing
from performance.spread import Spread
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from utils.argparse import parse_args
from utils.common import make_dir
from utils.dataset import load_dataset_from_openml
from utils.optimization import multi_objective, single_objective
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
    plot_pareto_from_smac,
    get_pareto_indicators,
)
from utils.sample import grid_search, random_search
from utils.input import ConfDict, create_configuration
from utils.output import (
    adapt_paretos,
    check_pictures,
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)


def calc_crowding_distance(F):
    infinity = 1e14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:
        # sort each column and get index
        I = np.argsort(F, axis=0, kind="mergesort")

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate(
            [np.full((1, n_obj), -np.inf), F]
        )

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = (
            np.sum(
                dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)],
                axis=1,
            )
            / n_obj
        )

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


A = np.array([[2, 0.5], [4, 0.6], [6, 0.7], [8, 0.8]])
B = np.array([[1, 0.5], [2, 0.6], [3, 0.7], [5, 0.8]])

ref_point = np.array([10, 0.5])
ideal_point = np.array([0, 0])

indHV = Hypervolume(ref_point=ref_point)
print("HV", indHV(A))
print("HV", indHV(B))
print()

indSP = Spacing()
print("SP", indSP(A))
print("SP", indSP(B))
print()

indMS = Spread(nadir=ref_point, ideal=ideal_point)
print("MS", indMS(A))
print("MS", indMS(B))
print()

indR2 = R2(ideal=ideal_point)
print("R2", indR2(A))
print("R2", indR2(B))
print()

args, _ = parse_args()
create_configuration(
    file_name=args.conf_file,
    origin="optimization",
)
indicators = get_pareto_indicators()
print(
    indicators["hv"]["indicator"](
        np.array(
            [
                [1 - elem[0], elem[1]]
                for elem in [
                    [0.7736345672607422, 0.3786627848943075],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                    [0.8181299591064453, 0.9016474088033041],
                ]
            ]
        )
    )
)

print(
    indicators["hv"]["indicator"](
        np.array(
            [
                [1 - elem[0], elem[1]]
                for elem in [
                    [0.665311279296875, 0.33026303847630817],
                    [0.7948455047607422, 1.7664515177408855],
                    [0.8007740020751953, 3.477466901143392],
                    [0.805676498413086, 4.966122309366861],
                    [0.813238525390625, 7.054574330647786],
                    [0.8215357208251953, 8.829967498779297],
                    [0.830468521118164, 11.140373229980469],
                    [0.8359800720214844, 13.284662882486979],
                    [0.8412681579589844, 16.03728993733724],
                    [0.8453408050537109, 18.528917948404946],
                ]
            ]
        )
    )
)
