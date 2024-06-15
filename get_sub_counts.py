import argparse
import types
from torch_geometric.datasets import TUDataset
import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.topology as gt_topology
import torch
from torch_geometric.utils import remove_self_loops, degree, to_undirected
from torch_geometric.data import Data
import os
import logging

def config_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    return logger


def subgraph_counts2ids(count_fn, data, subgraph_dicts, subgraph_params):

    if hasattr(data, 'edge_features'):
        edge_index, edge_features = remove_self_loops(data.edge_index, data.edge_features)
        setattr(data, 'edge_features', edge_features)
    else:
        edge_index = remove_self_loops(data.edge_index)[0]

    if data.x is None:
        num_nodes = data.edge_index.max().item() + 1
    else:
        num_nodes = data.x.shape[0]
    identifiers = None

    logger.info("num of subgraph_dicts: {}".format(len(subgraph_dicts)))
    for i, subgraph_dict in enumerate(subgraph_dicts):
        logger.info(i)
        kwargs = {'subgraph_dict': subgraph_dict,
                  'induced': subgraph_params['induced'],
                  'num_nodes': num_nodes,
                  'directed': subgraph_params['directed']}
        counts = count_fn(edge_index, **kwargs)
        identifiers = counts if identifiers is None else torch.cat((identifiers, counts), 1)
        
    setattr(data, 'edge_index', edge_index)
    setattr(data, 'identifiers', identifiers.long())

    return data


def automorphism_orbits(edge_list, print_msgs=True, **kwargs):

    directed = kwargs['directed'] if 'directed' in kwargs else False

    graph = gt.Graph(directed=directed)
    graph.add_edge_list(edge_list)
    gt.generation.remove_self_loops(graph)
    gt.generation.remove_parallel_edges(graph)

    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role

    orbit_membership_list = [[], []]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse=True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i, vertex in enumerate(orbit_membership_list[0])}

    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit] + [vertex]

    aut_count = len(aut_group)

    return graph, orbit_partition, orbit_membership, aut_count


def induced_edge_automorphism_orbits(edge_list, **kwargs):

    directed = kwargs['directed'] if 'directed' in kwargs else False
    directed_orbits = kwargs['directed_orbits'] if 'directed_orbits' in kwargs else False

    graph, orbit_partition, orbit_membership, aut_count = automorphism_orbits(edge_list=edge_list,
                                                                              directed=directed,
                                                                              print_msgs=False)
    edge_orbit_partition = dict()
    edge_orbit_membership = dict()
    edge_orbits2inds = dict()
    ind = 0

    if not directed:
        edge_list = to_undirected(torch.tensor(graph.get_edges()).transpose(1, 0)).transpose(1, 0).tolist()

    for i, edge in enumerate(edge_list):
        if directed_orbits:
            edge_orbit = (orbit_membership[edge[0]], orbit_membership[edge[1]])
        else:
            edge_orbit = frozenset([orbit_membership[edge[0]], orbit_membership[edge[1]]])
        if edge_orbit not in edge_orbits2inds:
            edge_orbits2inds[edge_orbit] = ind
            ind_edge_orbit = ind
            ind += 1
        else:
            ind_edge_orbit = edge_orbits2inds[edge_orbit]

        if ind_edge_orbit not in edge_orbit_partition:
            edge_orbit_partition[ind_edge_orbit] = [tuple(edge)]
        else:
            edge_orbit_partition[ind_edge_orbit] += [tuple(edge)]

        edge_orbit_membership[i] = ind_edge_orbit

    return graph, edge_orbit_partition, edge_orbit_membership, aut_count


def subgraph_isomorphism_vertex_counts(edge_index, **kwargs):

    subgraph_dict, induced, num_nodes = kwargs['subgraph_dict'], kwargs['induced'], kwargs['num_nodes']
    directed = kwargs['directed'] if 'directed' in kwargs else False

    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index.transpose(1, 0).cpu().numpy()))
    gt.generation.remove_self_loops(G_gt)
    gt.generation.remove_parallel_edges(G_gt)

    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True,
                                               generator=True)


    counts = np.zeros((num_nodes, len(subgraph_dict['orbit_partition'])))
    for sub_iso_curr in sub_iso:
        for i, node in enumerate(sub_iso_curr):
            counts[node, subgraph_dict['orbit_membership'][i]] += 1
    counts = counts / subgraph_dict['aut_count']

    counts = torch.tensor(counts)

    return counts


def subgraph_isomorphism_edge_counts(edge_index, **kwargs):

    subgraph_dict, induced = kwargs['subgraph_dict'], kwargs['induced']
    directed = kwargs['directed'] if 'directed' in kwargs else False

    edge_index = edge_index.transpose(1, 0).cpu().numpy()
    edge_dict = {}
    for i, edge in enumerate(edge_index):
        edge_dict[tuple(edge)] = i

    if not directed:
        subgraph_edges = to_undirected(
            torch.tensor(subgraph_dict['subgraph'].get_edges().tolist()).transpose(1, 0)).transpose(1, 0).tolist()

    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index))
    gt.stats.remove_self_loops(G_gt)
    gt.stats.remove_parallel_edges(G_gt)

    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True,
                                               generator=True)

    counts = np.zeros((edge_index.shape[0], len(subgraph_dict['orbit_partition'])))

    for sub_iso_curr in sub_iso:
        mapping = sub_iso_curr.get_array()
        for i, edge in enumerate(subgraph_edges):
            edge_orbit = subgraph_dict['orbit_membership'][i]
            mapped_edge = tuple([mapping[edge[0]], mapping[edge[1]]])
            counts[edge_dict[mapped_edge], edge_orbit] += 1

    counts = counts / subgraph_dict['aut_count']

    counts = torch.tensor(counts)

    return counts


def get_custom_edge_list(ks, substructure_type=None, filename=None):
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))
    return edge_lists


def main():
    k_max = args.k
    k_min = 1 if args.id_type == 'star_graph' else 2 
    custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), args.id_type)

    automorphism_fn = induced_edge_automorphism_orbits if args.id_scope == 'local' else automorphism_orbits
    count_fn = subgraph_isomorphism_edge_counts if args.id_scope == 'local' else subgraph_isomorphism_vertex_counts

    subgraph_params = {'induced': False,
                       'edge_list': custom_edge_list,
                       'directed': False,
                       'directed_orbits': False}

    subgraph_dicts = []
    orbit_partition_sizes = []
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
            automorphism_fn(edge_list=edge_list, directed=subgraph_params['directed'],
                            directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph': subgraph, 'orbit_partition': orbit_partition,
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))

        dataset = TUDataset('PretrainedDatasets', args.dataset)
        graphs_ptg = []
        for i, data in enumerate(dataset):
            ii = i + 1
            logger.info("graph index: {}".format(ii))
            new_data = data
            if new_data.edge_index.shape[1] == 0:
                continue
            else:
                setattr(new_data, 'degrees', degree(new_data.edge_index[0]))
            new_data = subgraph_counts2ids(count_fn, new_data, subgraph_dicts, subgraph_params)
            graphs_ptg.append(new_data)

        torch.save((graphs_ptg, orbit_partition_sizes), 'PretrainedDatasets/' + path + '.pt')
        print(graphs_ptg[0].x, graphs_ptg[0].edge_index, graphs_ptg[0].batch)

        new_data = dataset[0]
        if new_data.edge_index.shape[1] == 0:
            setattr(new_data, 'degrees', torch.zeros((new_data.graph_size,)))
        else:
            setattr(new_data, 'degrees', degree(new_data.edge_index[0]))
        new_data = subgraph_counts2ids(subgraph_isomorphism_vertex_counts, new_data, subgraph_dicts,
                                       subgraph_params)
        torch.save((new_data, orbit_partition_sizes), 'PretrainedDatasets/' + path + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Broad-Tweet-Corpus')
    parser.add_argument("--k", type=int, default=4,
                        help="count all the subgraphs of the family that have size up to k")
    parser.add_argument('--id_scope', type=str, default='global')
    parser.add_argument('--id_type', type=str, default='complete_graph')
    args = parser.parse_args()
    path = args.dataset + '_' + args.id_scope + '_' + args.id_type + '_' + str(args.k)
    logger = config_logger('logs_sub_counts/' + path + '.log')
    main()
