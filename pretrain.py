from torch_geometric import seed_everything
import warnings
import argparse
from pretrain import *
warnings.filterwarnings("ignore")

import pycuda.driver as cuda

cuda.init()
def get_parser():
    parser = argparse.ArgumentParser(description='SKIE Arguments.')

    # Key setting
    parser.add_argument("--method", type=str, default='RTGSN',
                        help="Use different contrastive learning methods. Can be chosen from "
                             "{'RTGSN', 'GSN'}, ")
    parser.add_argument("--dataset", type=str, default='NYT10-1')
    parser.add_argument("--feature", type=str, default='sub')
    parser.add_argument("--amr_dataset_file", type=str, default=None)
    parser.add_argument("--text_dataset_file", type=str, default=None)
    parser.add_argument("--bert_model_name", type=str, default='roberta-large')
    parser.add_argument("--tokenizer_name", type=str, default='roberta-large')

    # Basic setting
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--times", type=int, default=1,
                        help="The number of repetitions of the experiment.")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--cuda_device", type=int, default=1)


    # Dataset Preprocessing and augmentation
    parser.add_argument("--pn", type=float, default=0.2,
                        help="The probability of dropping node, removing edge, or sampling subgraph.")
    parser.add_argument("--factor", type=float, default=0.3) 
    parser.add_argument("--cal_weight", type=str, default='node')
    parser.add_argument("--core", type=str, default='kcore')
    parser.add_argument("--mode", type=str, default='T2G')
    parser.add_argument("--lossf", type=str, default='Triplet',
                       help="Contraste loss, can be chosen from {'InfoNCE', 'Triplet'}.")
    parser.add_argument("--margin", type=float, default=0.1,
                       help="Contrastive learning parameters, please make different adjustments for different contrastive learning methods.")

    # Model training
    parser.add_argument("--epoch", type=int, default=50,
                        help="Training epoch.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size of dataset partition.")
    parser.add_argument("--shuffle", type=bool, default=False,
                        help="Shuffle the graphs in the dataset or not.")
    parser.add_argument("--hid_units", type=int, default=64,
                        help="Dimension of hidden layers and embedding.")
    parser.add_argument("--num_layer", type=int, default=3,
                        help="Number of RTGSN layers.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of sampled graphs to train model.")

    # Model Saving and evaluation
    parser.add_argument("--interval", type=int, default=2,
                        help="Interval epoch to test.")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="Whether to save the model.")
    parser.add_argument("--save_embed", type=bool, default=True,
                        help="Whether to save the model.")
    parser.add_argument("--eval_model", type=bool, default=True,
                        help="Evaluate immediately or save the model.")
    parser.add_argument("--norm", type=bool, default=False,
                        help="Whether normalize embedding before logistic regression test.")
                        

    return parser


def arg_parse(parser0):
    args = parser0.parse_args()

    if torch.cuda.is_available():
        cuda_device_index = args.cuda_device
        args.device = torch.device(f"cuda:{cuda_device_index}")
    else:
        args.device = torch.device("cpu")
    

    if args.save_path is None:
        args.save_path = args.method + '_' + args.dataset + '_layer' + str(args.num_layer) + '_hid' + str(args.hid_units) + '_pn' + str(args.pn) + '_factor' + str(args.factor) + '_tau' + str(args.tau) + '_loff' + str(args.lossf) + '_epo' + str(args.epoch)
    if args.amr_dataset_file is None:
        args.amr_dataset_file = args.dataset + '_global_complete_graph_5.pt'
        args.text_dataset_file = args.dataset + '_text_oneie.json'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.save_path += '/' + args.amr_dataset_file.split('.')[0]
    return args


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = get_parser()
    args = arg_parse(parser)
    seed_everything(args.seed)
    method = Method(args)
    method.train()
    torch.cuda.empty_cache()

