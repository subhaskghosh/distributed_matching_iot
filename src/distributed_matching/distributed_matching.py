import math
import os
import random
import shutil
from typing import Dict, List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from math import ceil
###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def clean_output_folder(folder: str = "./output_graphs"):
    """Removes all files in the specified folder."""
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def plot_bipartite_graph_with_matching(G: nx.Graph, matching: List[Tuple], title: str = "Bipartite Graph with Matching"):
    """Plots the bipartite graph with the matching highlighted."""
    plt.figure(figsize=(10, 8))
    left_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    right_nodes = set(G.nodes()) - left_nodes
    pos = {}
    pos.update((node, (1, idx)) for idx, node in enumerate(left_nodes))
    pos.update((node, (2, idx)) for idx, node in enumerate(right_nodes))
    non_matching_edges = [edge for edge in G.edges() if edge not in matching and (edge[1], edge[0]) not in matching]
    nx.draw_networkx_edges(G, pos, edgelist=non_matching_edges, edge_color='blue', width=1)
    nx.draw_networkx_edges(G, pos, edgelist=matching, edge_color='red', width=2)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='white')
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)
    plt.plot([], [], color='red', linewidth=4, label='Matching Edges')
    plt.plot([], [], color='blue', linewidth=2, label='Non-matching Edges')
    plt.legend(loc='best')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./output_graphs/bipartite_graph_with_matching.png", dpi=300)
    plt.show()

###############################################################################
# (1) FRACTIONAL MATCHING STAGE
###############################################################################

def fractional_matching_algorithm(B: nx.Graph, approx_cover_threshold: float = 0.5, max_rounds: int = 1000) -> Tuple[Dict[tuple, float], Dict]:
    """
    Computes a fractional matching on B.
    Each edge gets weight = min(1/deg(u), 1/deg(v)) and nodes whose incident sum >= approx_cover_threshold are removed.
    Scales the maximum fractional weight to 0.5.
    Returns:
      - fraction_dict: mapping from edge to fractional weight
      - round_info: debug information per round
    """
    fraction_dict = {e: 0.0 for e in B.edges()}
    round_info = {}
    graph_current = B.copy()
    round_number = 0
    while graph_current.number_of_edges() > 0 and round_number < max_rounds:
        edges_round_weights = {(u, v): min(1.0 / graph_current.degree(u), 1.0 / graph_current.degree(v))
                                for u, v in graph_current.edges()}
        for e, w in edges_round_weights.items():
            fraction_dict[e] += w
        vertices_to_remove = [node for node in list(graph_current.nodes())
                              if sum(edges_round_weights.get((node, adj), 0.0) for adj in list(graph_current.neighbors(node))) >= approx_cover_threshold]
        graph_current.remove_nodes_from(vertices_to_remove)
        round_info[round_number] = {'edges_round_weights': edges_round_weights,
                                    'removed_vertices': vertices_to_remove,
                                    'graph_size': graph_current.number_of_edges()}
        round_number += 1
    # Scale fractional weights so that maximum is 0.5.
    if fraction_dict:
        max_weight = max(fraction_dict.values())
        if max_weight > 0:
            scale = 0.5 / max_weight
            for e in fraction_dict:
                fraction_dict[e] *= scale
    for e in fraction_dict:
        fraction_dict[e] = min(fraction_dict[e], 1.0)
    return fraction_dict, round_info

###############################################################################
# (2) TWO-POWER REDUCTION STAGE
###############################################################################

def two_power_reduction(fraction_dict: Dict, original_bip_G: nx.Graph) -> nx.MultiGraph:
    """
    Converts each fractional edge weight into a sum of powers of two.
    Builds a MultiGraph with parallel edges (each edge gets a weight that is a power of 2).
    """
    multiG = nx.MultiGraph()
    multiG.add_nodes_from(original_bip_G.nodes(data=True))
    for (u, v), frac_w in fraction_dict.items():
        if frac_w <= 0:
            continue
        remaining = frac_w
        tol = 1e-14
        power = 0.5  # since maximum fractional weight is scaled to 0.5
        while remaining > tol and power >= tol:
            if power <= remaining + tol:
                multiG.add_edge(u, v, weight=power, original_edge=(u, v))
                remaining -= power
            power /= 2.0
    return multiG

###############################################################################
# (3) GRADUAL ROUNDING STAGE (2-DECOMPOSITION + CONSISTENT COLORING)
###############################################################################

def two_decomposition(G: nx.MultiGraph, edges: List[Tuple]) -> nx.DiGraph:
    """
    Performs a 2-decomposition on the given set of edges from G.
    For each node in G, create copies and assign each edge uniquely.
    Returns a directed graph H with an attribute 'original_edge' on each edge.
    """
    H = nx.DiGraph()
    node_copies = {}
    for node in G.nodes():
        degree = G.degree(node)
        out_copies = [f"{node}_out_{i+1}" for i in range(degree)]
        in_copies = [f"{node}_in_{i+1}" for i in range(degree)]
        H.add_nodes_from(out_copies)
        H.add_nodes_from(in_copies)
        node_copies[node] = {"out": out_copies, "in": in_copies}
    for u, v in edges:
        try:
            assign_u = next(c for c in node_copies[u]["out"] if H.out_degree(c) < 1)
            assign_v = next(c for c in node_copies[v]["in"] if H.in_degree(c) < 1)
            H.add_edge(assign_u, assign_v, original_edge=(u, v))
        except StopIteration:
            raise ValueError(f"No valid copy found for edge ({u}, {v}).")
    return H

def reverse_path(G: nx.DiGraph, path: List[str], head: Dict, path_length: Dict):
    """
    Reverses the edges along a given path in the directed graph G.
    Updates the head and path_length dictionaries.
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            G.add_edge(v, u, **G.get_edge_data(u, v, default={}))
    head[path[-1]] = path[0]
    path_length[path[-1]] = len(path)

def consistent_coloring(G: nx.DiGraph, weight: float, phase: int) -> List[Tuple]:
    """
    Performs consistent coloring (and rounding) on the decomposed graph G.
    Uses iterative reversal of short paths to ensure alternating directions.
    Returns a list of original edges (as stored in attribute 'original_edge') from G.
    """
    l = 2 ** phase
    head = {node: node for node in list(G.nodes())}
    path_length = {node: 1 for node in list(G.nodes())}
    num_iters = ceil(math.log2(l))
    for _ in range(num_iters):
        # Iterate over a copy of node list to avoid modifying while iterating
        for v in list(G.nodes()):
            for _, w in list(G.out_edges(v)):
                try:
                    path_iter = nx.all_simple_paths(G, source=v, target=w, cutoff=l-1)
                    path = next(path_iter, None)
                    if path is not None and len(path) < l:
                        reverse_path(G, path, head, path_length)
                except nx.NetworkXNoPath:
                    continue
    rounded_edges = []
    for u, v in list(G.edges()):
        original_edge = G[u][v].get("original_edge")
        if original_edge and original_edge not in rounded_edges:
            rounded_edges.append(original_edge)
    return rounded_edges

def gradual_rounding_stage(multigraph: nx.MultiGraph, beta: float, epsilon: float) -> List[Tuple]:
    """
    Gradually rounds the fractional matching (represented as a multigraph) into an integral matching.
    Runs in phases; in each phase, selects edges of a specific power-of-two weight,
    applies 2-decomposition and consistent coloring, and removes the rounded edges.
    Returns the list of original edges that form the final matching.
    """
    matching = []
    delta_val = max(dict(multigraph.degree()).values()) if multigraph.nodes() else 0
    k = ceil(math.log2(delta_val / beta)) if delta_val > 0 else 1  # Number of phases
    for phase in range(1, k + 1):
        current_weight = 2 ** -(k - phase + 1)
        # Select edges with weight equal (within tolerance) to current_weight.
        phase_edges = [(u, v) for u, v, data in multigraph.edges(data=True)
                       if abs(data['weight'] - current_weight) < 1e-10]
        if not phase_edges:
            continue
        # Perform 2-decomposition on the selected edges.
        decomposed_graph = two_decomposition(multigraph, phase_edges)
        # Apply consistent coloring (which rounds these edges).
        rounded_edges = consistent_coloring(decomposed_graph, current_weight, phase)
        matching.extend(rounded_edges)
        # Remove from multigraph all edges whose original_edge appears in the rounded_edges.
        to_remove = []
        for u, v, key, data in list(multigraph.edges(data=True, keys=True)):
            orig = data.get("original_edge")
            if orig in rounded_edges or (orig is not None and orig[::-1] in rounded_edges):
                to_remove.append((u, v, key))
        multigraph.remove_edges_from(to_remove)
    return matching

###############################################################################
# (4) DISTRIBUTED MATCHING ALGORITHM
###############################################################################

def distributed_matching(G: nx.Graph) -> List[Tuple]:
    """
    Computes a (1-ε)-approximate maximum matching in bipartite graph G.
    Combines the fractional matching, two-power reduction, and gradual rounding stages.
    """
    gamma = 0.5    # Parameter for fractional matching
    beta = 0.5     # Parameter for rounding
    epsilon = 0.5  #
    frac_match, _ = fractional_matching_algorithm(G, approx_cover_threshold=0.5, max_rounds=500)
    multiG = two_power_reduction(frac_match, G)
    matching = gradual_rounding_stage(multiG, beta, epsilon)
    return matching

def distributed_max_weight_matching(G: nx.Graph, epsilon: float = 0.1) -> List[Tuple]:
    """
    Computes a (1/3 - ε)-approximate maximum weight matching in weighted bipartite graph G.
    Splits edge weights into geometric classes and calls distributed_matching on each residual graph.
    """
    beta = epsilon / 2
    W = max(data['weight'] for _, _, data in G.edges(data=True)) if G.edges() else 0
    delta = beta / 2
    k = ceil(math.log(W, 1 + delta)) if W > 0 else 0
    matchings = []
    # Process weight classes in descending order.
    for i in range(k):
        threshold = (1 + delta) ** (k - i - 1)
        edges = [(u, v, data['weight']) for u, v, data in G.edges(data=True) if data['weight'] >= threshold]
        if not edges:
            continue
        residual_G = nx.Graph()
        residual_G.add_weighted_edges_from(edges)
        # Remove nodes already matched from higher weight classes.
        if matchings:
            matched_nodes = set(u for m in matchings for (u, v) in m) | set(v for m in matchings for (u, v) in m)
            residual_G.remove_nodes_from(matched_nodes)
        if residual_G.number_of_edges() > 0:
            m = distributed_matching(residual_G)
            if m:
                matchings.append(m)
    final_matching = []
    matched_nodes = set()
    # Combine matchings from higher to lower classes.
    for matching in reversed(matchings):
        for u, v in matching:
            if u not in matched_nodes and v not in matched_nodes:
                final_matching.append((u, v))
                matched_nodes.update([u, v])
    return final_matching

###############################################################################
# (5) Example
###############################################################################

def generate_random_graph(n: int, p: float) -> nx.Graph:
    B = nx.bipartite.random_graph(n, n, p)
    new_edges = [(u, v, {"weight": random.randint(1, 700)}) for u, v in B.edges()]
    B.update(new_edges, B.nodes)
    return B

def main_validation():
    clean_output_folder()
    # Create a random weighted bipartite graph.
    G = generate_random_graph(2, 0.6)
    matching = distributed_max_weight_matching(G, epsilon=0.1)
    plot_bipartite_graph_with_matching(G, matching, "Maximum Weight Matching")
    print("Computed Maximum Weight Matching:", matching)

if __name__ == "__main__":
    main_validation()
