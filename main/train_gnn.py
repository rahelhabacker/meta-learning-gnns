import argparse
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data_prep.config import *
from data_prep.data_utils import SUPPORTED_DATASETS
from data_prep.data_utils import get_data
from models.gat_base import GatBase
from models.proto_maml import ProtoMAML
from models.proto_net import ProtoNet

SUPPORTED_MODELS = ['gat', 'prototypical', 'gmeta']
LOG_PATH = "../logs/"

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def train(model_name, seed, epochs, patience, h_size, top_users, top_users_excluded, k_shot, lr, lr_cl, lr_inner,
          lr_outer, hidden_dim, feat_reduce_dim, proto_dim, data_train, data_eval, dirs, checkpoint, train_docs,
          train_split_size, feature_type, vocab_size, n_inner_updates, num_workers, dropout, dropout_lin):
    os.makedirs(LOG_PATH, exist_ok=True)

    eval_split_size = (0.0, 0.25, 0.75) if data_eval != data_train else None

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    if checkpoint is not None and model_name not in checkpoint:
        raise ValueError(f"Can not evaluate model type '{model_name}' on a pretrained model of another type.")

    nr_train_docs = 'all' if (train_docs is None or train_docs == -1) else str(train_docs)

    # if we only want to evaluate, model should be initialized with nr of labels from evaluation data
    evaluation = checkpoint is not None and Path(checkpoint).exists()

    print(f'\nConfiguration:\n\n mode: {"TEST" if evaluation else "TRAIN"}\n seed: {seed}\n max epochs: {epochs}\n '
          f'patience:{patience}\n k_shot: {k_shot}\n\n model_name: {model_name}\n hidden_dim: {hidden_dim}\n '
          f'feat_reduce_dim: {feat_reduce_dim}\n checkpoint: {checkpoint}\n\n data_train: {data_train} '
          f'(splits: {str(train_split_size)})\n data_eval: {data_eval} (splits: {str(eval_split_size)})\n '
          f'nr_train_docs: {nr_train_docs}\n hop_size: {h_size}\n top_users: {top_users}K\n '
          f'top_users_excluded: {top_users_excluded}%\n num_workers: {num_workers}\n vocab_size: {vocab_size}\n '
          f'hops: {h_size}\n feature_type: {feature_type}\n\n lr: {lr}\n lr_cl: {lr_cl}\n outer_lr: {lr_outer}\n '
          f'inner_lr: {lr_inner}\n n_updates: {n_inner_updates}\n proto_dim: {proto_dim}\n')

    # reproducible results
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # the data preprocessing
    print('\nLoading data ..........')

    loaders, train_graph_size, eval_graph_size, labels, b_size, class_ratio, f1_targets = get_data(data_train,
                                                                                                   data_eval,
                                                                                                   model_name,
                                                                                                   h_size,
                                                                                                   top_users,
                                                                                                   top_users_excluded,
                                                                                                   k_shot,
                                                                                                   train_split_size,
                                                                                                   eval_split_size,
                                                                                                   feature_type,
                                                                                                   vocab_size,
                                                                                                   dirs,
                                                                                                   num_workers)

    optimizer_hparams = {
        "lr_cl": lr_cl,
        "lr": lr,
        'lr_inner': lr_inner,
        'lr_output': lr_outer
    }

    model_params = {
        'model': model_name,
        'hid_dim': hidden_dim,
        'feat_reduce_dim': feat_reduce_dim,
        'input_dim': train_graph_size[1],
        'output_dim': len(labels[0]),
        'proto_dim': proto_dim,
        'class_weight': class_ratio,
        'dropout': dropout,
        'dropout_lin': dropout_lin,
        'concat': True,
        'n_heads': 2
    }

    train_loader, train_val_loader, test_loader, test_val_loader = loaders

    # verify_not_overlapping_samples(train_loader)
    # verify_not_overlapping_samples(train_val_loader)
    # verify_not_overlapping_samples(test_val_loader)
    # verify_not_overlapping_samples(test_loader)

    print('\nInitializing trainer ..........\n')
    trainer = initialize_trainer(epochs, patience, model_name, lr, lr_cl, lr_inner, lr_outer, seed, data_train,
                                 data_eval, k_shot, h_size, feature_type, checkpoint)

    if model_name == 'gat':
        model = GatBase(model_params, optimizer_hparams, b_size, f1_targets[0], checkpoint)
    elif model_name == 'prototypical':
        model = ProtoNet(model_params['input_dim'], model_params['hid_dim'], model_params['feat_reduce_dim'],
                         optimizer_hparams['lr'], b_size, f1_targets[0])
    elif model_name == 'gmeta':
        model = ProtoMAML(model_params['input_dim'], model_params['hid_dim'], model_params['feat_reduce_dim'],
                          optimizer_hparams, n_inner_updates, b_size, f1_targets[0])
    else:
        raise ValueError(f'Model name {model_name} unknown!')

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
        model.reset_classifier_dimensions(len(labels[1]))

        # TODO: set also the target label for f1 score
        # f1_targets[1]

    evaluate(trainer, model, test_loader, test_val_loader)


def verify_not_overlapping_samples(train_val_loader):
    # support and query set should have same classes, but distinct examples
    first_n_equal, second_n_equal, not_different_examples = 0, 0, 0
    for sub_graphs, targets in iter(train_val_loader):
        chunked = targets.chunk(2, dim=0)
        bin_1 = torch.bincount(chunked[0])
        bin_2 = torch.bincount(chunked[1])
        # support and query should have same classes
        comp = bin_1 == bin_2
        sh = comp.shape[0]

        if sh == 1 and comp[0].item() is False:
            first_n_equal += 1

        if sh == 2 and comp[1].item() is False:
            second_n_equal += 1

        # different examples...
        center_indices = [s.orig_center_idx.item() for s in sub_graphs]
        not_different = len(center_indices) != len(set(center_indices))
        if not_different:
            not_different_examples += 1
    assert first_n_equal == 0
    assert second_n_equal == 0
    assert not_different_examples == 0


def initialize_trainer(epochs, patience, model_name, lr, lr_cl, lr_inner, lr_output, seed, data_train, data_eval,
                       k_shot, h_size, f_type, checkpoint):
    """
    Initializes a Lightning Trainer for respective parameters as given in the function header. Creates a proper
    folder name for the respective model files, initializes logging and early stopping.
    """

    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    base = f'dtrain={data_train}_deval={data_eval}_seed={seed}_shots={k_shot}_hops={h_size}_ftype={f_type}_lr={lr}'
    if model_name == 'gat':
        version_str = f'{base}_lr-cl={lr_cl}'
    elif model_name == 'prototypical':
        version_str = f'{base}'
    elif model_name == 'gmeta':
        version_str = f'{base}_lr-inner={lr_inner}_lr-output={lr_output}'
    else:
        raise ValueError(f'Model name {model_name} unknown!')

    logger = TensorBoardLogger(LOG_PATH, name=model_name, version=version_str)

    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=patience,  # validation happens per default after each training epoch
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(move_metrics_to_cpu=True,
                         log_every_n_steps=1,
                         logger=logger,
                         enable_checkpointing=True,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epochs,
                         callbacks=[model_checkpoint, early_stop_callback],
                         enable_progress_bar=True,
                         num_sanity_val_steps=0)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


def evaluate(trainer, model, test_dataloader, val_dataloader):
    """
    Tests a model on test and validation set.

    Args:
        trainer (pl.Trainer) - Lightning trainer to use.
        model (pl.LightningModule) - The Lightning Module which should be used.
        test_dataloader (DataLoader) - Data loader for the test split.
        val_dataloader (DataLoader) - Data loader for the validation split.
    Returns:
        test_accuracy (float) - The achieved test accuracy.
        val_accuracy (float) - The achieved validation accuracy.
    """

    print('\nTesting model on validation and test ..........\n')

    test_start = time.time()

    results = trainer.test(model, dataloaders=[test_dataloader, val_dataloader], verbose=False)

    test_results = results[0]
    val_results = results[1]

    test_accuracy = test_results['test_accuracy/dataloader_idx_0']
    test_f1 = test_results['test_f1']
    test_f1_macro = test_results['test_f1_macro/dataloader_idx_0']
    test_f1_micro = test_results['test_f1_micro/dataloader_idx_0']

    val_accuracy = val_results['test_accuracy/dataloader_idx_1']
    val_f1 = val_results['test_val_f1']
    val_f1_macro = val_results['test_f1_macro/dataloader_idx_1']
    val_f1_micro = val_results['test_f1_micro/dataloader_idx_1']

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n '
          f'test accuracy: {round(test_accuracy, 3)} ({test_accuracy})\n '
          f'test f1: {round(test_f1, 3)} ({test_f1})\n '
          f'test f1 micro: {round(test_f1_micro, 3)} ({test_f1_micro})\n '
          f'test f1 macro: {round(test_f1_macro, 3)} ({test_f1_macro})\n '
          f'validation accuracy: {round(val_accuracy, 3)} ({val_accuracy})\n '
          f'validation f1: {round(val_f1, 3)} ({val_f1})\n '
          f'validation f1 micro: {round(val_f1_micro, 3)} ({val_f1_micro})\n '
          f'validation f1 macro: {round(val_f1_macro, 3)} ({val_f1_macro})\n '
          f'\nepochs: {trainer.current_epoch + 1}\n')

    print(f'{round_format(test_f1)}\n{round_format(test_f1_micro)}\n{round_format(test_f1_macro)}\n'
          f'{round_format(test_accuracy)}\n{round_format(val_f1)}\n{round_format(val_f1_micro)}\n'
          f'{round_format(val_f1_macro)}\n{round_format(val_accuracy)}')

    return test_accuracy, val_accuracy


def round_format(metric):
    # locale.setlocale(locale.LC_ALL, 'de_DE')
    # return locale.format_string('%.3f', round(metric, 3), grouping=True)
    return f"{round(metric, 3):%.3f}".replace(".", ",")


if __name__ == "__main__":
    # tsv_dir = TSV_small_DIR
    # complete_dir = COMPLETE_small_DIR
    # num_nodes = int(COMPLETE_small_DIR.split('-')[1])

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

    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1)
    parser.add_argument('--dropout-linear', dest='dropout_lin', type=float, default=0.5)
    parser.add_argument('--k-shot', dest='k_shot', type=int, default=20, help="Number of examples per task/batch.")
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--lr-cl', dest='lr_cl', type=float, default=0.001,
                        help="Classifier learning rate for baseline.")

    # MODEL CONFIGURATION

    parser.add_argument('--model', dest='model', default='gat', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=512)
    parser.add_argument('--feature-reduce-dim', dest='feat_reduce_dim', type=int, default=10000)
    parser.add_argument('--checkpoint', default=model_checkpoint, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: None)')

    # META PARAMETERS

    parser.add_argument('--proto-dim', dest='proto_dim', type=int, default=64)
    parser.add_argument('--outer-lr', dest='lr_outer', type=float, default=0.01)
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

    params = vars(parser.parse_args())

    train(
        model_name=params['model'],
        seed=params['seed'],
        epochs=params['epochs'],
        patience=params['patience'],
        h_size=params["hop_size"],
        top_users=params["top_users"],
        top_users_excluded=params["top_users_excluded"],
        k_shot=params["k_shot"],
        lr=params["lr"],
        lr_cl=params["lr_cl"],
        lr_inner=params["lr_inner"],
        lr_outer=params["lr_outer"],
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
        dropout=params["dropout"],
        dropout_lin=params["dropout_lin"]
    )
