import argparse
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data_prep.config import TSV_DIR, COMPLETE_DIR
from data_prep.data_utils import get_data, SUPPORTED_DATASETS
from models.gat_base import GatBase
from models.proto_maml import ProtoMAML
from models.proto_net import ProtoNet, test_proto_net
from samplers.batch_sampler import SHOTS

SUPPORTED_MODELS = ['gat', 'prototypical', 'gmeta']
LOG_PATH = "../logs/"

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def train(progress_bar, model_name, seed, epochs, patience, patience_metric, h_size, top_users, top_users_excluded,
          k_shot, lr, lr_cl, lr_inner, lr_output, hidden_dim, feat_reduce_dim, proto_dim, data_train, data_eval,
          dirs, checkpoint, train_docs, train_split_size, feature_type, vocab_size, n_inner_updates, num_workers,
          gat_dropout, lin_dropout, attn_dropout, wb_mode, warmup, max_iters, gat_heads, gat_batch_size):
    os.makedirs(LOG_PATH, exist_ok=True)

    eval_split_size = (0.0, 0.25, 0.75) if data_eval != data_train else None

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    if checkpoint is not None and model_name not in checkpoint:
        raise ValueError(f"Can not evaluate model type '{model_name}' on a pretrained model of another type.")

    if k_shot not in SHOTS:
        raise ValueError(f"'{k_shot}' is not valid!")

    nr_train_docs = 'all' if (train_docs is None or train_docs == -1) else str(train_docs)

    # if we only want to evaluate, model should be initialized with nr of labels from evaluation data
    evaluation = checkpoint is not None and Path(checkpoint).exists()

    print(f'\nConfiguration:\n\n mode: {"TEST" if evaluation else "TRAIN"}\n seed: {seed}\n max epochs: {epochs}\n '
          f'patience: {patience}\n patience metric: {patience_metric}\n k_shot: {k_shot}\n\n model_name: {model_name}\n'
          f' hidden_dim: {hidden_dim}\n feat_reduce_dim: {feat_reduce_dim}\n checkpoint: {checkpoint}\n '
          f' gat heads: {gat_heads}\n\n'
          f' data_train: {data_train} (splits: {str(train_split_size)})\n data_eval: {data_eval} '
          f'(splits: {str(eval_split_size)})\n nr_train_docs: {nr_train_docs}\n hop_size: {h_size}\n '
          f'top_users: {top_users}K\n top_users_excluded: {top_users_excluded}%\n num_workers: {num_workers}\n '
          f'vocab_size: {vocab_size}\n feature_type: {feature_type}\n\n lr: {lr}\n lr_cl: {lr_cl}\n '
          f'lr_output: {lr_output}\n inner_lr: {lr_inner}\n n_updates: {n_inner_updates}\n proto_dim: {proto_dim}\n')

    # reproducible results
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # the data preprocessing
    print('\nLoading data ..........')

    loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                top_users_excluded, k_shot, train_split_size, eval_split_size,
                                                feature_type, vocab_size, dirs, gat_batch_size, num_workers)

    train_loader, train_val_loader, test_loader, test_val_loader = loaders

    optimizer_hparams = {"lr": lr, "warmup": warmup,
                         "max_iters": len(train_loader) * epochs if max_iters < 0 else max_iters}

    model_params = {
        'model': model_name,
        'hid_dim': hidden_dim,
        'feat_reduce_dim': feat_reduce_dim,
        'input_dim': train_graph.size[1],
        'output_dim': len(train_graph.labels),
        'class_weight': train_graph.class_ratio,
        'gat_dropout': gat_dropout,
        'lin_dropout': lin_dropout,
        'attn_dropout': attn_dropout,
        'concat': True,
        'n_heads': gat_heads
    }

    if model_name == 'gat':
        optimizer_hparams.update(lr_cl=lr_cl)

        model = GatBase(model_params, optimizer_hparams, train_graph.label_names, train_loader.b_size)
    elif model_name == 'prototypical':
        model_params.update(proto_dim=proto_dim)

        model = ProtoNet(model_params, optimizer_hparams, train_graph.label_names, train_loader.b_size)
    elif model_name == 'gmeta':
        model_params.update(n_inner_updates=n_inner_updates)
        optimizer_hparams.update(lr_output=lr_output, lr_inner=lr_inner)

        model = ProtoMAML(model_params, optimizer_hparams, train_graph.label_names, train_loader.b_size)
    else:
        raise ValueError(f'Model name {model_name} unknown!')

    print('\nInitializing trainer ..........\n')

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
        batch_sizes=dict(train=train_loader.b_size, val=train_val_loader.b_size, test=test_loader.b_size),
        num_batches=dict(train=len(train_loader), val=len(train_val_loader), test=len(test_loader)),
        train_splits=dict(train=train_split_size[0], val=train_split_size[1], test=train_split_size[2]),
        nr_train_docs=nr_train_docs,
        top_users=top_users,
        top_users_excluded=top_users_excluded,
        num_workers=num_workers,
        vocab_size=vocab_size
    )

    trainer = initialize_trainer(epochs, patience, patience_metric, data_train, progress_bar, wb_mode, wandb_config)

    if not evaluation:
        # Training

        print('\nFitting model ..........\n')
        start = time.time()
        trainer.fit(model, train_loader, train_val_loader)

        end = time.time()
        elapsed = end - start
        print(f'\nRequired time for training: {int(elapsed / 60)} minutes.\n')

        # Load the best checkpoint after training
        model_path = trainer.checkpoint_callback.best_model_path
        print(f'Best model path: {model_path}')
    else:
        model_path = checkpoint

    # Evaluation
    model = model.load_from_checkpoint(model_path)

    # model was trained on another dataset --> reinitialize gat classifier
    if model_name == 'gat' and data_eval is not None and data_eval != data_train:
        # TODO: this is completely newly setting the output layer, erases all pretrained weights!
        model.reset_classifier_dimensions(len(eval_graph.labels))
        # f1_train_label, _ = train_graph.f1_target_label, eval_graph.f1_target_label
        # TODO: set also the target label for f1 score
        # f1_targets[1]

    test_accuracy, val_accuracy, val_f1_fake, val_f1_real, val_f1_macro = 0.0, 0.0, 0.0, 0.0, 0.0

    if model_name == 'gat':
        test_f1_fake, test_f1_real, test_f1_macro, val_f1_fake, val_f1_real, val_f1_macro, test_elapsed \
            = evaluate(trainer, model, test_loader, test_val_loader)

    elif model_name == 'prototypical':
        (test_f1_fake, _), (test_f1_real, _), (test_f1_macro, _), test_elapsed, _ \
            = test_proto_net(model, eval_graph, len(eval_graph.labels), data_feats=None, k_shot=k_shot)
    else:
        return

    wandb.log({
        "test/f1_fake": test_f1_fake,
        "test/f1_real": test_f1_real,
        "test/f1_macro": test_f1_macro,
        "test_val/f1_fake": val_f1_fake,
        "test_val/f1_real": val_f1_real,
        "test_val/f1_macro": val_f1_macro
    })

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n '
          f'test f1 fake: {round(test_f1_fake, 3)} ({test_f1_fake})\n '
          f'test f1 real: {round(test_f1_real, 3)} ({test_f1_real})\n '
          f'test f1 macro: {round(test_f1_macro, 3)} ({test_f1_macro})\n '
          f'validation f1 fake: {round(val_f1_fake, 3)} ({val_f1_fake})\n '
          f'validation f1 real: {round(val_f1_real, 3)} ({val_f1_real})\n '
          f'validation f1 macro: {round(val_f1_macro, 3)} ({val_f1_macro})\n '
          f'\nepochs: {trainer.current_epoch + 1}\n')

    print(f'{trainer.current_epoch + 1}\n{get_epoch_num(model_path)}\n{round_format(test_f1_fake)}\n'
          f'{round_format(test_f1_real)}\n{round_format(test_f1_macro)}\n{round_format(test_accuracy)}\n'
          f'{round_format(val_f1_fake)}\n{round_format(val_f1_real)}\n{round_format(val_f1_macro)}\n'
          f'{round_format(val_accuracy)}\n')


def get_epoch_num(model_path):
    epoch_str = 'epoch='
    start_idx = model_path.find(epoch_str) + len(epoch_str)
    expected_epoch = model_path[start_idx: start_idx + 2]
    if expected_epoch.endswith('-'):
        expected_epoch = expected_epoch[:1]
    return int(expected_epoch)


def initialize_trainer(epochs, patience, patience_metric, data_train, progress_bar, wb_mode, wandb_config):
    """
    Initializes a Lightning Trainer for respective parameters as given in the function header. Creates a proper
    folder name for the respective model files, initializes logging and early stopping.
    """

    if patience_metric == 'loss':
        cls, metric, mode = LossEarlyStopping, 'val/loss', 'min'
    elif patience_metric == 'f1_macro':
        cls, metric, mode = EarlyStopping, 'val/f1_macro', 'max'
    else:
        raise ValueError(f"Patience metric '{patience_metric}' is not supported.")

    logger = WandbLogger(project='meta-gnn',
                         name=f"{time.strftime('%Y%m%d_%H%M', time.gmtime())}_{data_train}",
                         log_model=True if wb_mode == 'online' else False,
                         save_dir=LOG_PATH,
                         offline=wb_mode == 'offline',
                         config=wandb_config)

    early_stop_callback = cls(
        monitor=metric,
        min_delta=0.00,
        patience=patience,  # loss computation happens per default after each training epoch
        verbose=False,
        mode=mode
    )

    mc_callback = cb.ModelCheckpoint(save_weights_only=True, mode=mode, monitor=metric)

    trainer = pl.Trainer(move_metrics_to_cpu=True,
                         log_every_n_steps=1,
                         logger=logger,
                         enable_checkpointing=True,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epochs,
                         callbacks=[mc_callback, early_stop_callback],
                         enable_progress_bar=progress_bar,
                         num_sanity_val_steps=0)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


class LossEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, _):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


def evaluate(trainer, model, test_dataloader, val_dataloader):
    """
    Tests a model on test and validation set.

    Args:
        trainer (pl.Trainer) - Lightning trainer to use.
        model (pl.LightningModule) - The Lightning Module which should be used.
        test_dataloader (DataLoader) - Data loader for the test split.
        val_dataloader (DataLoader) - Data loader for the validation split.
    """

    print('\nTesting model on validation and test ..........\n')

    test_start = time.time()

    results = trainer.test(model, dataloaders=[test_dataloader, val_dataloader], verbose=False)

    test_results = results[0]
    val_results = results[1]

    test_f1_fake = test_results['test/f1_fake']
    test_f1_real = test_results['test/f1_real']
    test_f1_macro = test_results['test/f1_macro']

    val_f1_fake = val_results['test/f1_fake']
    val_f1_real = val_results['test/f1_real']
    val_f1_macro = test_results['test/f1_macro']

    test_end = time.time()
    test_elapsed = test_end - test_start

    return test_f1_fake, test_f1_real, test_f1_macro, val_f1_fake, val_f1_real, val_f1_macro, test_elapsed


def round_format(metric):
    return f"{round(metric, 3):.3f}".replace(".", ",")


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
    parser.add_argument('--epochs', dest='epochs', type=int, default=1)
    parser.add_argument('--patience-metric', dest='patience_metric', type=str, default='loss')
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

    params = vars(parser.parse_args())

    os.environ["WANDB_MODE"] = params['wb_mode']

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
        gat_batch_size=params['gat_batch_size']
    )
