import argparse
import random
import sys
import warnings

from data_prep.config import TSV_DIR, COMPLETE_DIR
from samplers.batch_sampler import SHOTS
from train_gnn import SUPPORTED_MODELS, LOG_PATH

warnings.filterwarnings("ignore")
sys.path.append("..")
# from torchsummary import summary

import nltk

from main.data_prep.data_utils import get_data, SUPPORTED_DATASETS

nltk.download('punkt')

from train_gnn_main import *
from caching_funcs.cache_gnn import *


def train(progress_bar, model_name, seed, epochs, patience, patience_metric, h_size, top_users, top_users_excluded,
          k_shot, lr, lr_cl, lr_inner, lr_output, hidden_dim, feat_reduce_dim, proto_dim, data_train, data_eval,
          dirs, checkpoint, train_docs, train_split_size, feature_type, vocab_size, n_inner_updates, num_workers,
          gat_dropout, lin_dropout, attn_dropout, wb_mode, warmup, max_iters, gat_heads, gat_batch_size, loss_func,
          fc_dim, dropout, node_drop, scheduler, optimizer, weight_decay, pos_wt, lr_decay_step, lr_decay_factor):
    # if not os.path.exists(config['data_path']):
    #     raise ValueError("[!] ERROR: Dataset path does not exist")
    # else:
    #     print("\nData path checked..")

    # if not os.path.exists(config['model_path']):
    #     print("\nCreating checkpoint path for saved models at:  {}\n".format(config['model_path']))
    #     os.makedirs(config['model_path'])
    # else:
    #     print("\nModel save path checked..")

    # if config['model_name'] not in ['gcn', 'graph_sage', 'graph_conv', 'gat', 'rgcn', 'rgat', 'HGCN', 'HNN']:
    #     raise ValueError(
    #         "[!] ERROR:  model_name is incorrect. Choose one of - gcn / graph_sage / graph_conv / gat / rgcn / rgat / HGCN / HNN")
    # else:
    #     print("\nModel name checked...")

    # if not os.path.exists(config['vis_path']):
    #     print("\nCreating checkpoint path for Tensorboard visualizations at:  {}\n".format(config['vis_path']))
    #     os.makedirs(config['vis_path'])
    # else:
    #     print("\nTensorbaord Visualization path checked..")
    #     print("Cleaning Visualization path of older tensorboard files...\n")
    #     # shutil.rmtree(config['vis_path'])

    # Seeds for reproducible runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare dataset and iterators for training
    # prep_data = Prepare_GNN_Dataset(config)
    # config['loader'], config['vocab_size'], config['data'] = prep_data.prepare_gnn_training(verbose=False)

    # data_train = 'gossipcop'
    # data_eval = 'gossipcop'
    # model_name = 'gat'
    # h_size = 2
    # top_users_excluded = 1
    # top_users = 30
    # k_shot = 5
    # train_split_size = (0.7, 0.1, 0.2)
    # eval_split_size = (0.7, 0.1, 0.2)
    # feature_type = 'one-hot'
    # vocab_size = 10000
    # gat_batch_size = 344
    # dirs = ('data', TSV_DIR, COMPLETE_DIR)

    eval_split_size = train_split_size

    loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                top_users_excluded, k_shot, train_split_size, eval_split_size,
                                                feature_type, vocab_size, dirs, gat_batch_size, num_workers=0)

    # Setting up logging
    wandb_config = dict(
        seed=seed,
        max_epochs=epochs,
        patience=patience,
        patience_metric=patience_metric,
        k_shot=k_shot,
        h_size=h_size,
        checkpoint=checkpoint,
        data_train=data_train,
        data_eval=data_eval,
        feature_type=feature_type,
        batch_sizes=dict(train=loaders[0].b_size, val=loaders[1].b_size, test=loaders[2].b_size),
        num_batches=dict(train=len(loaders[0]), val=len(loaders[1]), test=len(loaders[2])),
        train_splits=dict(train=train_split_size[0], val=train_split_size[1], test=train_split_size[2]),
        top_users=top_users,
        top_users_excluded=top_users_excluded,
        num_workers=num_workers,
        vocab_size=vocab_size
    )

    wandb.init(name=f"{time.strftime('%Y%m%d_%H%M', time.gmtime())}_{data_train}", project='meta-gnn',
               entity='rahelhabacker', dir=LOG_PATH, reinit=True,
               config=wandb_config, group='foo-group', job_type='foo-job-type')

    model_config = dict(vocab_size=vocab_size, data_name=data_train, max_epoch=epochs, patience=patience,
                        patience_metric=patience_metric, loss_func=loss_func, embed_dim=hidden_dim, dropout=dropout,
                        fc_dim=fc_dim, node_drop=node_drop, model_name=model_name, n_classes=len(train_graph.labels),
                        scheduler=scheduler, optimizer=optimizer, lr=lr, weight_decay=weight_decay, pos_wt=pos_wt,
                        lr_decay_step=lr_decay_step, lr_decay_factor=lr_decay_factor)

    try:
        graph_net = GraphNetMain(model_config, loaders, vocab_size)
        graph_net.train_main()

    except KeyboardInterrupt:
        print("Keyboard interrupt by user detected...\nClosing the tensorboard writer!")
        print("Best val f1 = ", graph_net.best_val_f1)
        wandb.finish()


if __name__ == "__main__":
    # tsv_dir = TSV_small_DIR
    # complete_dir = COMPLETE_small_DIR
    # num_nodes = int(COMPLETE_small_DIR.split('-')[1])

    # model_checkpoint = '../logs/prototypical/dtrain=gossipcop_deval=gossipcop_seed=1234_shots=5_hops=2_ftype=one-hot_lr=0.0001/checkpoints/epoch=1-step=709-v1.ckpt'
    # model_checkpoint = '../logs/gat/dtrain=gossipcop_deval=None_seed=82_shots=2_hops=2_ftype=one-hot_lr=0.0001_lr-cl=0.001/checkpoints/epoch=16-step=27488.ckpt'
    # model_checkpoint = '../logs/prototypical/dname=gossipcop_seed=1234_lr=0.01/checkpoints/epoch=0-step=8-v4.ckpt'
    model_checkpoint = None

    tsv_dir = TSV_DIR
    complete_dir = COMPLETE_DIR
    num_nodes = -1

    # MAML setup
    # proto_dim = 64,
    # lr = 1e-3,
    # lr_inner = 0.1,
    # lr_output = 0.1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--progress-bar', dest='progress_bar', action='store_true')
    parser.add_argument('--no-progress-bar', dest='progress_bar', action='store_false')
    parser.set_defaults(progress_bar=True)

    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--epochs', dest='epochs', type=int, default=5)
    parser.add_argument('--patience-metric', dest='patience_metric', type=str, default='f1')
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--gat-dropout', dest='gat_dropout', type=float, default=0.6)
    parser.add_argument('--lin-dropout', dest='lin_dropout', type=float, default=0.5)
    parser.add_argument('--attn-dropout', dest='attn_dropout', type=float, default=0.6)
    parser.add_argument('--k-shot', dest='k_shot', type=int, default=5, help="Number of examples per task/batch.",
                        choices=SHOTS)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--lr-cl', dest='lr_cl', type=float, default=0.001,
                        help="Classifier learning rate for baseline.")
    parser.add_argument("--warmup", dest='warmup', type=int, default=500,
                        help="Number of steps for which we do learning rate warmup.")
    parser.add_argument("--max-iters", dest='max_iters', type=int, default=-1,
                        help='Number of iterations until the learning rate decay after warmup should last. '
                             'If not given then it is computed from the given epochs.')

    # MODEL CONFIGURATION

    parser.add_argument('--model', dest='model', default='gat', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=512)
    parser.add_argument('--gat-heads', dest='gat_heads', type=int, default=2)
    parser.add_argument('--feature-reduce-dim', dest='feat_reduce_dim', type=int, default=256)
    parser.add_argument('--checkpoint', default=model_checkpoint, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: None)')

    # META PARAMETERS

    parser.add_argument('--proto-dim', dest='proto_dim', type=int, default=64)
    parser.add_argument('--output-lr', dest='lr_output', type=float, default=0.01)
    parser.add_argument('--inner-lr', dest='lr_inner', type=float, default=0.01)
    parser.add_argument('--n-updates', dest='n_updates', type=int, default=5,
                        help="Inner gradient updates during meta learning.")

    # DATA CONFIGURATION
    parser.add_argument('--hop-size', dest='hop_size', type=int, default=2)
    parser.add_argument('--top-users', dest='top_users', type=int, default=30)
    parser.add_argument('--top-users-excluded', type=int, default=1,
                        help='Percentage (in %) of top sharing users that are excluded (the bot users).')
    parser.add_argument('--n-workers', dest='n_workers', type=int, default=None,
                        help="Amount of parallel data loaders.")
    parser.add_argument('--dataset-train', dest='dataset_train', default='gossipcop', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use for training. '
                             'If a checkpoint is provided we do not train again.')
    parser.add_argument('--dataset-eval', dest='dataset_eval', default='gossipcop', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use for evaluation.')
    parser.add_argument('--num-train-docs', dest='num_train_docs', type=int, default=num_nodes,
                        help="Inner gradient updates during meta learning.")
    parser.add_argument('--feature-type', dest='feature_type', type=str, default='one-hot',
                        help="Type of features used.")
    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument('--gat-batch-size', dest='gat_batch_size', type=int, default=344,
                        help="Size of batches for the GAT baseline.")
    parser.add_argument('--data-dir', dest='data_dir', default='data',
                        help='Select the dataset you want to use.')

    # parser.add_argument('--train-size', dest='train_size', type=float, default=0.875)
    # parser.add_argument('--val-size', dest='val_size', type=float, default=0.125)
    # parser.add_argument('--test-size', dest='test_size', type=float, default=0.0)
    parser.add_argument('--train-size', dest='train_size', type=float, default=0.7)
    parser.add_argument('--val-size', dest='val_size', type=float, default=0.1)
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2)

    parser.add_argument('--tsv-dir', dest='tsv_dir', default=tsv_dir,
                        help='Select the dataset you want to use.')
    parser.add_argument('--complete-dir', dest='complete_dir', default=complete_dir,
                        help='Select the dataset you want to use.')

    parser.add_argument('--wb-mode', dest='wb_mode', type=str, default='offline')

    parser.add_argument('--loss_func', type=str, default='bce_logits',
                        help='Loss function to use for optimization: bce / bce_logits / ce')

    parser.add_argument('--fc_dim', type=int, default=64, help='dimension of hidden layers of the MLP classifier')

    parser.add_argument('--dropout', type=float, default=0.2, help='Regularization - dropout on hidden embeddings')

    parser.add_argument('--node_drop', type=float, default=0.2, help='Node dropout to drop entire node from a batch')

    parser.add_argument('--optimizer', type=str, default='RAdam', help='Optimizer to use for training')

    parser.add_argument('--scheduler', type=str, default='step',
                        help='The type of lr scheduler to use anneal learning rate: step/multi_step')

    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for optimizer')

    parser.add_argument('--pos_wt', type=float, default=3,
                        help='Loss reweighing for the positive class to deal with class imbalance')

    parser.add_argument('--lr_decay_step', type=float, default=5,
                        help='No. of epochs after which learning rate should be decreased')

    parser.add_argument('--lr_decay_factor', type=float, default=0.8,
                        help='Decay the learning rate of the optimizer by this multiplicative amount')

    params = vars(parser.parse_args())

    # os.environ["WANDB_MODE"] = params['wb_mode']
    os.environ["WANDB_MODE"] = 'online'

    # Print args
    print("\n" + "x" * 50 + "\n\nRunning training with the following parameters: \n")
    for key, value in params.items():
        print(key + ' : ' + str(value))
    print("\n" + "x" * 50)

    train(
        progress_bar=params['progress_bar'],
        model_name=params['model'],
        seed=params['seed'],
        epochs=params['epochs'],
        patience=params['patience'],
        patience_metric=params['patience_metric'],
        h_size=params["hop_size"],
        top_users=params["top_users"],
        top_users_excluded=params["top_users_excluded"],
        k_shot=params["k_shot"],
        lr=params["lr"],
        lr_cl=params["lr_cl"],
        lr_inner=params["lr_inner"],
        lr_output=params["lr_output"],
        hidden_dim=params["hidden_dim"],
        feat_reduce_dim=params["feat_reduce_dim"],
        proto_dim=params["proto_dim"],
        data_train=params["dataset_train"],
        data_eval=params["dataset_eval"],
        dirs=(params["data_dir"], params["tsv_dir"], params["complete_dir"]),
        checkpoint=params["checkpoint"],
        train_docs=params["num_train_docs"],
        train_split_size=(params["train_size"], params["val_size"], params["test_size"]),
        feature_type=params["feature_type"],
        vocab_size=params["vocab_size"],
        n_inner_updates=params["n_updates"],
        num_workers=params["n_workers"],
        gat_dropout=params["gat_dropout"],
        lin_dropout=params["lin_dropout"],
        attn_dropout=params["attn_dropout"],
        wb_mode=params['wb_mode'],
        warmup=params['warmup'],
        max_iters=params['max_iters'],
        gat_heads=params['gat_heads'],
        gat_batch_size=params['gat_batch_size'],
        loss_func=params['loss_func'],
        fc_dim=params['fc_dim'],
        dropout=params['dropout'],
        node_drop=params['node_drop'],
        scheduler=params['scheduler'],
        optimizer=params['optimizer'],
        weight_decay=params['weight_decay'],
        pos_wt=params['pos_wt'],
        lr_decay_step=params['lr_decay_step'],
        lr_decay_factor=params['lr_decay_factor']
    )
