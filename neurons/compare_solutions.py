import asyncio
import json
import random
import time
import numpy as np
import bittensor as bt
from pydantic import ValidationError

from graphite.data import ASIA_MSB_DETAILS, load_dataset, WORLD_TSP_DETAILS
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
from graphite.protocol import GraphV2Synapse, GraphV2Problem
from neurons.call_method import beam_solver_solution, baseline_solution, nns_vali_solver_solution, hpn_solver_solution, \
    scoring_solution, tsp_annealer_solver

loaded_datasets = {
    ASIA_MSB_DETAILS['ref_id']: load_dataset(ASIA_MSB_DETAILS['ref_id']),
    WORLD_TSP_DETAILS['ref_id']: load_dataset(WORLD_TSP_DETAILS['ref_id'])
}


def generate_problem_from_dataset(min_node=2000, max_node=5000):
    n_nodes = random.randint(min_node, max_node)
    # randomly select n_nodes indexes from the selected graph
    prob_select = random.randint(0, len(list(loaded_datasets.keys()))-1)
    dataset_ref = list(loaded_datasets.keys())[prob_select]
    bt.logging.info(f"n_nodes V2 {n_nodes}")
    bt.logging.info(f"dataset ref {dataset_ref} selected from {list(loaded_datasets.keys())}" )
    bt.logging.info(f"dataset length {len(loaded_datasets[dataset_ref]['data'])} from {loaded_datasets[dataset_ref]['data'].shape} " )
    selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]['data'])), n_nodes)
    test_problem_obj = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)

    try:
        graphsynapse_req = GraphV2Synapse(problem=test_problem_obj)
        bt.logging.info(f"GraphV2Synapse Problem, n_nodes: {graphsynapse_req.problem.n_nodes}")
    except ValidationError as e:
        bt.logging.debug(f"GraphV2Synapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)
    return graphsynapse_req


def recreate_edges(problem: GraphV2Problem):
    node_coords_np = loaded_datasets[problem.dataset_ref]["data"]
    node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
    if problem.cost_function == "Geom":
        return geom_edges(node_coords)
    elif problem.cost_function == "Euclidean2D":
        return euc_2d_edges(node_coords)
    elif problem.cost_function == "Manhatten2D":
        return man_2d_edges(node_coords)
    else:
        return "Only Geom, Euclidean2D, and Manhatten2D supported for now."


def compare(min_node=2000, max_node=5000):
    synapse_request = generate_problem_from_dataset(min_node=min_node,max_node=max_node)
    edges = recreate_edges(synapse_request.problem)
    synapse_request.problem.edges = edges

    t1 = time.time_ns()
    baseline_synapse = asyncio.run(baseline_solution(synapse_request))
    t2 = time.time_ns()
    annealer_synapse = asyncio.run(tsp_annealer_solver(synapse_request))
    t3 = time.time_ns()

    print(
        f'Computational time: {(t2 - t1) / 1e6} ms, time annealer: {(t3 - t2) / 1e6}, number of nodes: {synapse_request.problem.n_nodes}')
    list_synapse = [baseline_synapse, annealer_synapse]
    scores = [scoring_solution(synapse) for synapse in list_synapse]

    min_score = min(scores)
    scores.append(min_score)


    return scores



if __name__ == '__main__':
    ...
