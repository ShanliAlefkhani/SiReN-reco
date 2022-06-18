import torch
from evaluator import evaluator as ev
from util import gen_top_k
from dataloader import SiReNDataLoader
import argparse
from torch_geometric.data import Data
from sgcn import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    data_class = SiReNDataLoader(args.dataset, args.version)
    print('data loading...')
    sum_eval_p = [0,0,0,0,0]
    sum_eval_r = [0,0,0,0,0]
    sum_eval_n = [0,0,0,0,0]
    iteration = 1
    for i in range(iteration):
        train, test = data_class.data_load()
        train = train.astype({'userId': 'int64', 'movieId': 'int64'})
        offset = args.offset

        edge_user = train[train['rating'] > offset]['userId'].values - 1
        edge_item = train[train['rating'] > offset]['movieId'].values - 1 + data_class.num_u

        edge_user_n = train[train['rating'] <= offset]['userId'].values - 1
        edge_item_n = train[train['rating'] <= offset]['movieId'].values - 1 + data_class.num_u

        data_p = [[edge_user[i],edge_item[i]] for i in range(len(edge_user))]
        data_n = [[edge_user_n[i],edge_item_n[i]] for i in range(len(edge_user_n))]
        #edge_p = torch.stack((torch.cat((edge_user, edge_item), 0), torch.cat((edge_item, edge_user), 0)), 0)
        #data_p = Data(edge_index=edge_p)

        #edge_n = torch.stack((torch.cat((edge_user_n, edge_item_n), 0), torch.cat((edge_item_n, edge_user_n), 0)), 0)
        #data_n = Data(edge_index=edge_n)

        args2 = parameter_parser()
        tab_printer(args2)
        edges2 = {
            "positive_edges": data_p,
            "negative_edges": data_n,
            "ecount": 0,
            "ncount": data_class.num_u + data_class.num_v,
        }
        trainer = SignedGCNTrainer(args2, edges2)
        trainer.setup_dataset()
        z_list = trainer.create_and_train_model()

        data_class.train = train
        data_class.test = test
        diagram_data = []
        for EPOCH in range(len(z_list)):
            emb = z_list[EPOCH]
            emb_u, emb_v = torch.split(emb, [data_class.num_u, data_class.num_v])
            emb_u = emb_u.cpu().detach()
            emb_v = emb_v.cpu().detach()
            r_hat = emb_u.mm(emb_v.t())
            reco = gen_top_k(data_class, r_hat)
            eval_ = ev(data_class, reco, args)
            eval_.precision_and_recall()
            eval_.normalized_DCG()
            file = open("output.txt", "a")
            file.write(f"""***************************************************************************************
            /* Recommendation Accuracy */
            N :: {eval_.N}
            Precision at :: {eval_.N}, {eval_.p['total'][eval_.N - 1]}
            Recall at [10, 15, 20] :: {eval_.r['total'][eval_.N - 1]}
            nDCG at [10, 15, 20] :: {eval_.nDCG['total'][eval_.N - 1]}
            ***************************************************************************************""")
            diagram_data.append((EPOCH, eval_.nDCG['total'][eval_.N - 1]))

        plt.close()
        for j in range(5):
            xdata = np.asarray([x[0] for x in diagram_data])
            ydata = np.asarray([x[1][j] for x in diagram_data])
            plt.plot(xdata, ydata, 'o')
        plt.savefig(f'nDCG.png')
        sum_eval_p += eval_.p['total'][eval_.N - 1]
        sum_eval_r += eval_.r['total'][eval_.N - 1]
        sum_eval_n += eval_.nDCG['total'][eval_.N - 1]
    sum_eval_p = [x/iteration for x in sum_eval_p]
    sum_eval_r = [x/iteration for x in sum_eval_r]
    sum_eval_n = [x/iteration for x in sum_eval_n]

    file.write("Mean of Precision: {}".format(sum_eval_p))
    file.write("Mean of Recall: {}".format(sum_eval_r))
    file.write("Mean of nDCG: {}".format(sum_eval_n))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='ML-100K',
                        help="Dataset"
                        )
    parser.add_argument('--version',
                        type=int,
                        default=1,
                        help="Dataset version"
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help="Batch size"
                        )

    parser.add_argument('--dim',
                        type=int,
                        default=64,
                        help="Dimension"
                        )
    parser.add_argument('--lr',
                        type=float,
                        default=5e-3,
                        help="Learning rate"
                        )
    parser.add_argument('--offset',
                        type=float,
                        default=3.5,
                        help="Criterion of likes/dislikes"
                        )
    parser.add_argument('--K',
                        type=int,
                        default=40,
                        help="The number of negative samples"
                        )
    parser.add_argument('--num_layers',
                        type=int,
                        default=4,
                        help="The number of layers of a GNN model for the graph with positive edges"
                        )
    parser.add_argument('--MLP_layers',
                        type=int,
                        default=2,
                        help="The number of layers of MLP for the graph with negative edges"
                        )
    parser.add_argument('--epoch',
                        type=int,
                        default=20,
                        help="The number of epochs"
                        )
    parser.add_argument('--reg',
                        type=float,
                        default=0.05,
                        help="Regularization coefficient"
                        )
    arguments = parser.parse_args()
    main(arguments)
