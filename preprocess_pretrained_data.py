import torch
import dgl
import json
import os
from amr_parser.stack_transformer_amr_parser import AMRParser
import numpy as np

device = torch.device("cpu")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
def get_amr_edge_idx(edge_type_str):
    if edge_type_str in ['location', 'destination', 'path']:
        return 0
    elif edge_type_str in ['year', 'time', 'duration', 'decade', 'weekday']:
        return 1
    elif edge_type_str in ['instrument', 'manner', 'poss', 'topic', 'medium', 'duration']:
        return 2
    elif edge_type_str in ['mod']:
        return 3
    elif edge_type_str.startswith('prep-'):
        return 4
    elif edge_type_str.startswith('op') and edge_type_str[-1].isdigit():
        return 5
    elif edge_type_str == 'ARG0':
        return 6
    elif edge_type_str == 'ARG1':
        return 7
    elif edge_type_str == 'ARG2':
        return 8
    elif edge_type_str == 'ARG3':
        return 9
    elif edge_type_str == 'ARG4':
        return 10
    else:
        return 11

def amr_parse(tokens_list, output_dir):
    parser = AMRParser.from_checkpoint('./amr_general/checkpoint_best.pt')
    tokens_list = tokens_list
    amr_list = parser.parse_sentences(tokens_list)
    torch.save(amr_list, output_dir)

def processing_amr(amr_dir, tokens_list):
    amr_list = torch.load(amr_dir, map_location=device)

    node_idx_list, edge_type_list, node_idx_offset_list, node_idx_offset_whole = [], [], [], [] 
    list_of_align_dict = []  
    list_of_exist_dict = []  

    total_edge_num = 0 
    covered_edge_num = 0  
    order_list = []
    for i, amr in enumerate(amr_list):
        amr_split_list = amr.split('\n')
        node_to_idx, node_to_offset, node_to_offset_whole = {}, {}, {}
        node_num = 0
        for line in amr_split_list:
            if line.startswith('# ::node'):
                node_split = line.split('\t')
                if len(node_split) != 4:
                    continue
                else:
                    align_span = node_split[3].split('-')
                    if not align_span[0].isdigit():
                        continue
                    head_word_idx = int(align_span[1]) - 1
                    try:
                        start = int(align_span[0])
                    except:
                        raise ValueError
                    end = int(align_span[1])
                    if (start, end) not in list(node_to_offset_whole.values()):
                        node_to_offset.update({node_split[1]: head_word_idx})
                        node_to_offset_whole.update({node_split[1]: (start, end)})
                        node_to_idx.update({node_split[1]: node_num})
                        node_num += 1
            else:
                continue

        node_idx_list.append(node_to_idx)
        node_idx_to_offset = {}
        for key in node_to_idx.keys():
            node_idx_to_offset.update({node_to_idx[key]: node_to_offset[key]})

        node_idx_to_offset_whole = {}
        for key in node_to_idx.keys():
            node_idx_to_offset_whole.update({node_to_idx[key]: node_to_offset_whole[key]})

        node_idx_offset_list.append(node_idx_to_offset)
        node_idx_offset_whole.append(node_idx_to_offset_whole)
        edge_type_dict = {}

        for line in amr_split_list:
            if line.startswith('# ::root'):
                root_split = line.split('\t')
                root = root_split[1]
        prior_dict = {root:[]}

        start_list = []
        end_list = []

        for line in amr_split_list:
            if line.startswith('# ::edge'):
                edge_split = line.split('\t')
                amr_edge_type = edge_split[2]
                edge_start = edge_split[4]
                edge_end = edge_split[5]
                if (edge_start in node_to_idx) and (edge_end in node_to_idx):
                    if amr_edge_type.startswith("ARG") and amr_edge_type.endswith("-of"):
                        edge_start, edge_end = edge_end, edge_start
                        amr_edge_type = amr_edge_type[0:4]
                    edge_idx = get_amr_edge_idx(amr_edge_type)
                    total_edge_num += 1
                    if edge_idx == 11:
                        covered_edge_num += 1
                    start_idx = node_to_idx[edge_start]
                    end_idx = node_to_idx[edge_end]
                    edge_type_dict.update({(start_idx, end_idx): edge_idx})
                
                else:
                    continue
                if edge_end != root and (not ((edge_start in end_list) and (edge_end in start_list))):
                    start_list.append(edge_start)
                    end_list.append(edge_end)
                if edge_start not in prior_dict:
                    prior_dict.update({edge_start:[edge_end]})
                else:
                    prior_dict[edge_start].append(edge_end)
            else:
                continue
        edge_type_list.append(edge_type_dict)
        final_order_list = []
        candidate_nodes = node_to_idx.copy()
        while len(candidate_nodes) != 0:
            current_level_nodes = []
            for key in candidate_nodes:
                if key not in end_list:
                    final_order_list.append(candidate_nodes[key])
                    current_level_nodes.append(key)
            for node in current_level_nodes:
                candidate_nodes.pop(node)
            
            for node in current_level_nodes:
                indices_list = [i for i, x in enumerate(start_list) if x == node]
                start_list = [x for x in start_list if x != node]
                new_end_list = []
                for i in range(len(end_list)):
                    if i not in indices_list:
                        new_end_list.append(end_list[i])
                end_list = new_end_list

        order_list.append(final_order_list.copy())
    graphs_list = []

    for i in range(len(node_idx_list)):
        graph_i = dgl.DGLGraph()
        graph_i = graph_i.to(device)

        edge2type = edge_type_list[i]
        node2offset = node_idx_offset_list[i]
        node2offset_whole = node_idx_offset_whole[i]

        nodes_num = len(node2offset)

        graph_i.add_nodes(nodes_num)
        graph_i.ndata['token_pos'] = torch.zeros(nodes_num, 1, dtype=torch.long, device=device)
        graph_i.ndata['token_span'] = torch.zeros(nodes_num, 2, dtype=torch.long, device=device)

        for key in node2offset:
            graph_i.ndata['token_pos'][key][0] = node2offset[key]
        for key in node2offset:
            graph_i.ndata['token_span'][key][0] = node2offset_whole[key][0]
            graph_i.ndata['token_span'][key][1] = node2offset_whole[key][1]
        node_prior_tensor = torch.zeros(nodes_num, 1, dtype=torch.long)
        for j in range(nodes_num):
            node_prior_tensor[j][0] = order_list[i].index(j)
        graph_i.ndata['priority'] = node_prior_tensor.to(device)
        edge_num = len(edge2type)
    
        edge_iter = 0
        edge_type_tensor = torch.zeros(2 * edge_num, 1, dtype=torch.long)
        for key in edge2type:
            graph_i.add_edges(key[0], key[1])
            edge_type_tensor[edge_iter][0] = edge2type[key]
            edge_iter += 1

        for key in edge2type:
            graph_i.add_edges(key[1], key[0])
            edge_type_tensor[edge_iter][0] = edge2type[key]
            edge_iter += 1
        
        graph_i.edata['type'] = edge_type_tensor.to(device)
        graphs_list.append(graph_i)

        align_dict = {}
        exist_dict = {}

        span_list = graph_i.ndata["token_span"].tolist()

        for p in range(len(tokens_list[i])):
            min_dis = 2 * len(tokens_list[i])
            min_dis_idx = -1

            if_found = 0

            for q in range(len(span_list)):
                if p >= span_list[q][0] and p < span_list[q][1]:
                    if_found = 1
                    align_dict.update({p: q})
                    exist_dict.update({p: 1})
                    break
                else:
                    new_dis_1 = abs(p - span_list[q][0])
                    new_dis_2 = abs(p - (span_list[q][1] - 1))
                    new_dis = min(new_dis_1, new_dis_2)
                    if new_dis < min_dis:
                        min_dis = new_dis
                        min_dis_idx = q
            
            if not if_found:
                align_dict.update({p: min_dis_idx})
                exist_dict.update({p: 0})

        list_of_align_dict.append(align_dict)
        list_of_exist_dict.append(exist_dict)
    return graphs_list, list_of_align_dict, list_of_exist_dict


def get_amr_data(json_path, graph_pkl_path, amr_path):
    with open(json_path, "r", encoding='utf-8') as f:
        sents = [json.loads(line)['tokens'] for line in f if line.strip()]

    processed_sents = preprocess_sentences(sents)
    amr_parse(processed_sents, amr_path)
    graphs, align, exist = processing_amr(amr_path, processed_sents)
    torch.save((graphs, align, exist), graph_pkl_path)


def process_and_save_graph_data(json_data_path, graphs_pkl_path, output_edges_txt_path, output_indicator_txt_path, output_graph_label_txt_path, output_oneie_json_path):
    graphs, _, _ = torch.load(graphs_pkl_path, map_location=device)
    with open(json_data_path, 'r', encoding='utf-8') as file:
        json_data = [line for line in file]

    all_edges_list = []
    graph_indicator = []
    graph_labels = []
    global_node_id = 1 
    updated_json_data = []

    for graph, json_item in zip(graphs, json_data):
        if graph.number_of_nodes() < 3:
            continue  
        updated_json_data.append(json_item)  
        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_simple(graph)

        degrees = graph.in_degrees() + graph.out_degrees()
        non_isolated_nodes = torch.nonzero(degrees > 0, as_tuple=False).squeeze()

        local_to_global = {local_id.item(): global_id for local_id, global_id in zip(non_isolated_nodes, range(global_node_id, global_node_id + len(non_isolated_nodes)))}

        global_node_id += len(non_isolated_nodes)

        for start, end in zip(*graph.edges()):
            if start.item() in local_to_global and end.item() in local_to_global:
                updated_start = local_to_global[start.item()]
                updated_end = local_to_global[end.item()]
                all_edges_list.append([updated_start, updated_end])

        graph_indicator.extend([len(graph_labels) + 1] * len(non_isolated_nodes))
        graph_labels.append(1) 

    edges_array = np.array(all_edges_list)
    np.savetxt(output_edges_txt_path, edges_array, fmt='%d', delimiter=', ')

    with open(output_indicator_txt_path, 'w') as file:
        for indicator in graph_indicator:
            file.write(f"{indicator}\n")

    with open(output_graph_label_txt_path, 'w') as file:
        for label in graph_labels:
            file.write(f"{label}\n")

    with open(output_oneie_json_path, 'w', encoding='utf-8') as file:
        for item in updated_json_data:
            file.write(item)


def preprocess_sentences(sentences, max_length=510):
    processed_sentences = []
    for sentence in sentences:
        if len(sentence) > max_length:
            parts = [sentence[i:i + max_length -1] for i in range(0, len(sentence), max_length)]
            processed_sentences.extend(parts)
        else:
            processed_sentences.append(sentence)
    return processed_sentences


def run_pipeline(data_name, data_dir):
    json_data_path = os.path.join(data_dir, f"train.json")

    raw_data_dir = os.path.join(data_dir, "./raw")

    graphs_pkl_path = os.path.join(data_dir, f"train_graphs.pkl")
    amr_pkl_path = os.path.join(data_dir, f"{data_name}_amrs.pkl")

    output_edges_txt_path = os.path.join(raw_data_dir, f"{data_name}_A.txt")
    output_indicator_txt_path = os.path.join(raw_data_dir, f"{data_name}_graph_indicator.txt")
    output_graph_label_txt_path = os.path.join(raw_data_dir, f"{data_name}_graph_label.txt")

    output_oneie_json_path = os.path.join(data_dir, f"{data_name}_text_oneie.json")
    
    get_amr_data(json_data_path, graphs_pkl_path, amr_pkl_path)
    process_and_save_graph_data(json_data_path, graphs_pkl_path, output_edges_txt_path, output_indicator_txt_path, output_graph_label_txt_path, output_oneie_json_path)


def process_dataset(data_name, base_data_dir="./PretrainedDatasets"):
    data_dir = os.path.join(base_data_dir, data_name)
    if os.path.isdir(data_dir):
        print(f"Processing dataset: {data_name}")
        run_pipeline(data_name, data_dir)
    else:
        print(f"Directory {data_dir} does not exist.")



if __name__ == "__main__":
    dataset_name = 'Broad-Tweet-Corpus'
    process_dataset(dataset_name)