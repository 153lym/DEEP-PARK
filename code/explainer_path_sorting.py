import argparse
import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm

from utils import set_seed, print_args, set_config_args
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
from explainer import PaGELink


def read_dict(file_path):
    f_read = open(file_path, 'rb')
    data = pickle.load(f_read)
    f_read.close()
    return data


def retrieve_dict(original_dict):
    reversed_dict = {v: k for k, v in original_dict.items()}
    return reversed_dict


def load_args():
    parser = argparse.ArgumentParser(description='Explain link predictor')
    parser.add_argument('--device_id', type=int, default=-1)

    '''
    Dataset args
    '''
    parser.add_argument('--source_dir', type=str, default='..')
    parser.add_argument('--dataset_dir', type=str, default='/datasets')
    parser.add_argument('--graph_node_dir', type=str, default='/data/graph/nodes/')
    parser.add_argument('--drug_disease_file', type=str, default='/data/case/example.xlsx')
    parser.add_argument('--results_dir', type=str, default='/results/')

    parser.add_argument('--dataset_name', type=str, default='graph')
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--max_num_samples', type=int, default=-1,
                        help='maximum number of samples to explain, for fast testing. Use all if -1')

    '''
    GNN args
    '''
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--saved_model_dir', type=str, default='/code/saved_models')
    parser.add_argument('--saved_model_name', type=str, default='graph_model')

    '''
    Link predictor args
    '''
    parser.add_argument('--src_ntype', type=str, default='drug', help='prediction source node type')
    parser.add_argument('--tgt_ntype', type=str, default='disease', help='prediction target node type')
    parser.add_argument('--pred_etype', type=str, default='treats', help='prediction edge type')
    parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                        help='operation passed to dgl.EdgePredictor')

    '''
    Explanation args
    '''
    parser.add_argument('--lr', type=float, default=0.01, help='explainer learning_rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='explainer on-path edge regularizer weight')
    parser.add_argument('--beta', type=float, default=1.0, help='explainer off-path edge regularizer weight')
    parser.add_argument('--num_hops', type=int, default=5, help='computation graph number of hops')
    parser.add_argument('--num_epochs', type=int, default=50, help='How many epochs to learn the mask')
    parser.add_argument('--num_paths', type=int, default=200, help='How many paths to generate')
    parser.add_argument('--max_path_length', type=int, default=5, help='max lenght of generated paths')
    parser.add_argument('--k_core', type=int, default=2, help='k for the k-core graph')
    parser.add_argument('--prune_max_degree', type=int, default=-1,
                        help='prune the graph such that all nodes have degree smaller than max_degree. No prune if -1')

    args = parser.parse_args()
    return args


def load_trained_model(args):
    args.src_ntype = 'drug'
    args.tgt_ntype = 'disease'
    args.pred_etype = 'treats'

    if torch.cuda.is_available() and args.device_id >= 0:
        device = torch.device('cuda', index=args.device_id)
    else:
        device = torch.device('cpu')

    if args.link_pred_op in ['cat']:
        pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
    else:
        pred_kwargs = {}

    # print_args(args)
    set_seed(0)

    processed_g = load_dataset(args.source_dir + args.dataset_dir, args.dataset_name, args.pred_etype, args.valid_ratio,
                               args.test_ratio,
                               eval_exp=False)[1]
    mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g.to(device) for g in processed_g]

    encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
    model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
    state = torch.load(f'{args.source_dir + args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
    model.load_state_dict(state)

    pagelink = PaGELink(model,
                        lr=args.lr,
                        alpha=args.alpha,
                        beta=args.beta,
                        num_epochs=args.num_epochs,
                        log=True).to(device)

    return model, pagelink, mp_g


def generate_nids(args):
    nodes_path = args.source_dir + args.graph_node_dir
    node_icd = read_dict(nodes_path + 'icd11_dict.pkl')
    node_drug = read_dict(nodes_path + 'drug_dict.pkl')

    drugs_nids = []
    diseases_nids = []
    data = pd.read_excel(args.source_dir + args.drug_disease_file)
    drug_name = data['drug'].tolist()
    disease_name = data['disease'].tolist()
    disease_icd = data['ICD-11'].tolist()

    for dg in drug_name:
        drugs_nids.append(node_drug[dg])

    # add target-disease
    disease_name_icd_dict = {}
    for dis, dicd in zip(disease_name, disease_icd):
        query_dis = '[ICD-11: %s]' % dicd.replace(' ', '')
        diseases_nids.append(node_icd[query_dis])
        disease_name_icd_dict[query_dis] = dis

    query_src_nids, query_tgt_nids = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    for tgt_nid, src_nid in zip(diseases_nids, drugs_nids):
        query_src_nids = torch.cat((query_src_nids, torch.tensor([src_nid])), dim=0)
        query_tgt_nids = torch.cat((query_tgt_nids, torch.tensor([tgt_nid])), dim=0)

    return query_src_nids, query_tgt_nids, disease_name_icd_dict


def path_sorting(args, model, pagelink, mp_g, query_src_nids, query_tgt_nids):
    test_ids = range(query_src_nids.shape[0])
    # if args.max_num_samples > 0:
    #     test_ids = test_ids[:args.max_num_samples]

    pred_edge_to_comp_g_edge_mask = {}
    pred_edge_to_paths = {}
    for i in tqdm(test_ids):
        src_nid, tgt_nid = query_src_nids[i].unsqueeze(0), query_tgt_nids[i].unsqueeze(0)

        with torch.no_grad():
            pred = model(src_nid, tgt_nid, mp_g).sigmoid().item() > 0.5

        if pred:
            src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
            paths, comp_g_edge_mask_dict = pagelink.explain(src_nid,
                                                            tgt_nid,
                                                            mp_g,
                                                            args.num_hops,
                                                            args.prune_max_degree,
                                                            args.k_core,
                                                            args.num_paths,
                                                            args.max_path_length,
                                                            return_mask=True)

            pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict
            pred_edge_to_paths[src_tgt] = paths
    return pred_edge_to_paths


def transfer_output(data, disease_name_icd_dict):
    node_dict = {}
    nodes_path = args.source_dir + args.graph_node_dir
    node_dict['disease'] = retrieve_dict(read_dict(nodes_path + 'icd11_dict.pkl'))
    node_dict['pathway'] = retrieve_dict(read_dict(nodes_path + 'pathway_dict.pkl'))
    node_dict['drug'] = retrieve_dict(read_dict(nodes_path + 'drug_dict.pkl'))
    node_dict['target'] = retrieve_dict(read_dict(nodes_path + 'target_dict.pkl'))
    node_dict['target1'] = retrieve_dict(read_dict(nodes_path + 'target_dict.pkl'))
    node_dict['target2'] = retrieve_dict(read_dict(nodes_path + 'target_dict.pkl'))

    case = data.keys()
    SaveList = []
    for dis in case:
        (src_type, src_nid), (tgt_type, tgt_nid) = dis
        src = node_dict[src_type][src_nid]
        tgt = node_dict[tgt_type][tgt_nid]
        # pair = (src, tgt)
        # SaveList.append(str(pair))
        i = 1
        for paths in data[dis]:
            line = []
            for path in paths:
                edge, src_nid, tgt_nid = path
                src_type, edge_type, tgt_type = edge
                src = node_dict[src_type][src_nid]
                tgt = node_dict[tgt_type][tgt_nid]
                if tgt_type == 'disease':
                    tgt = disease_name_icd_dict[tgt]
                #     tgt = icd2dis[tgt]

                if src_type == 'drug' and len(line) == 0:
                    # line += [src, edge_type, tgt]
                    line += [src, tgt]
                else:
                    # line += [edge_type, tgt]
                    line += [tgt]
            # print(str(line))
            line = [i] + line
            SaveList.append(line)
            i += 1

    if not os.path.exists(args.source_dir + args.results_dir):
        os.makedirs(args.source_dir + args.results_dir)
    pd.DataFrame(SaveList).to_csv(args.source_dir + args.results_dir + 'pred_path_sorting_list.csv',
                                  header=['Ranking', 'drug', 'drug-target', 'KEGG pathway', 'disease-target', 'disease'])
    return SaveList


if __name__ == '__main__':
    # load the parameters
    args = load_args()

    # load the trained model
    model, pagelink_model, mp_g = load_trained_model(args)

    # Generate the query pair nids
    query_src_nids, query_tgt_nids, disease_name_icd_dict = generate_nids(args)

    # path sorting of the drug-disease pair
    pred_edge_to_paths = path_sorting(args, model, pagelink_model, mp_g, query_src_nids, query_tgt_nids)

    # transfer output
    pred_path_sorting_list = transfer_output(pred_edge_to_paths, disease_name_icd_dict)
