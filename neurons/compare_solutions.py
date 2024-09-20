import asyncio
import json
import random
import time
import numpy as np
import bittensor as bt
from pydantic import ValidationError

from graphite.data import ASIA_MSB_DETAILS, load_dataset, WORLD_TSP_DETAILS
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
from graphite.solvers.christofides import solve
from graphite.protocol import GraphV2Synapse, GraphV2Problem
from neurons.call_method import beam_solver_solution, baseline_solution, nns_vali_solver_solution, hpn_solver_solution, \
    scoring_solution, enhanced_solver_solution, or_solver_solution

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
    t1 = time.time()
    beam_synapse = asyncio.run(beam_solver_solution(synapse_request))
    t2 = time.time()
    baseline_synapse = asyncio.run(baseline_solution(synapse_request))
    t3 = time.time()
    nns_vali_synapse = asyncio.run(nns_vali_solver_solution(synapse_request))
    t4 = time.time()
    hpn_synapse = asyncio.run(hpn_solver_solution(synapse_request))
    t5 = time.time()
    christ_synapse = solve(synapse_request)
    t6 = time.time()
    enhanced_synapse = asyncio.run(enhanced_solver_solution(synapse_request))
    t7 = time.time()
    or_synapse = asyncio.run(or_solver_solution(synapse_request))
    t8 = time.time()
    d1 = t2 - t1
    d2 = t3 - t2
    d3 = t4 - t3
    d4 = t5 - t4
    d5 = t6 - t5
    d6 = t7 - t5
    d7 = t8 - t7
    if d1 > 10 :
        print(f"d1")
        exit(1)
    if d2 > 10 :
        print(f"d2")
        exit(1)
    if d3 > 10 :
        print(f"d3")
        exit(1)
    if d4 > 10 :
        print(f"d4")
        exit(1)
    if d5 > 10 :
        print(f"d5")
        exit(1)
    if d6 > 10 :
        print(f"d6")
    if d7 > 10 :
        print(f"d7")       


    list_synapse = [beam_synapse, baseline_synapse,nns_vali_synapse,hpn_synapse,christ_synapse,enhanced_synapse,or_synapse]
    scores = [scoring_solution(synapse) for synapse in list_synapse]

    min_score = min(scores)  # this give inacurrate result when any result is inf
    scores.append(min_score)

    return scores


if __name__ == '__main__':
    # synapse_request = generate_problem()
    # # print(f"synapse_request = {synapse_request}")
    # json_data = json.dumps(synapse_request.problem.dict())
    # print(f"synapse_request problem = {json_data}")
    # graph_problem_instance = GraphProblem.parse_raw(json_data)
    # print(f"GraphProblem instance: {isinstance(graph_problem_instance, GraphProblem)}")

    # synapse = asyncio.run(beam_solver_solution(synapse_request))
    # print(f"route = {synapse.solution}  length = {len(synapse.solution)}")
    # score = scoring_solution(synapse)
    # print(f"score = {score}")
    synapse_request = generate_problem()
    synapse = solve(synapse_request)
    print(f"tsp_tour = {synapse.solution}")
