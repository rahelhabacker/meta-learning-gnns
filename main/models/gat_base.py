import torch.nn.functional
import torch.nn.functional as func
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from models.graph_trainer import GraphTrainer
from models.gat_encoder_sparse_pushkar import GatNet
from models.train_utils import *


class GatBase(GraphTrainer):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, label_names, batch_size):
        """
        Args:
            model_params - Hyperparameters for the whole model, as dictionary.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__(model_params["output_dim"])

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace + saves config in wandb
        self.save_hyperparameters()

        self.model = GatNet(model_params)

        # flipping the weights
        pos_weight = 1 // model_params["class_weight"][0]
        # flipped_weights = torch.flip(fake_class_weight, dims=[0])
        self.loss_module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def configure_optimizers(self):
        train_optimizer, train_scheduler = self.get_optimizer()
        return [train_optimizer], [train_scheduler]

    def get_optimizer(self, model=None):
        opt_params = self.hparams.optimizer_hparams

        model = self.model if model is None else model

        if opt_params['optimizer'] == 'Adam':
            optimizer = AdamW(model.parameters(), lr=opt_params['lr'], weight_decay=opt_params['weight_decay'])
        elif opt_params['optimizer'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=opt_params['lr'], momentum=opt_params['momentum'],
                            weight_decay=opt_params['weight_decay'])
        else:
            raise ValueError("No optimizer name provided!")

        scheduler = None
        if opt_params['scheduler'] == 'step':
            scheduler = StepLR(optimizer, step_size=opt_params['lr_decay_epochs'],
                               gamma=opt_params['lr_decay_factor'])
        elif opt_params['scheduler'] == 'multi_step':
            scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 30, 40, 55],
                                    gamma=opt_params['lr_decay_factor'])

        return optimizer, scheduler

    def forward(self, sub_graphs, targets, mode=None):

        # make a batch out of all sub graphs and push the batch through the model
        x, edge_index, cl_mask = get_subgraph_batch(sub_graphs)

        logits = self.model(x, edge_index, mode)[cl_mask].squeeze()

        # make probabilities out of logits via sigmoid --> especially for the metrics; makes it more interpretable
        predictions = torch.sigmoid(logits).argmax(dim=-1)

        for mode_dict, _ in self.metrics.values():
            # shapes should be: pred (batch_size), targets: (batch_size)
            # print(f"Preds shape: {predictions.shape}")
            # print(f"Preds type: {type(predictions)}")
            # print(f"Targets shape: {targets.shape}")
            # print(f"Targets type: {type(targets)}")
            mode_dict[mode].update(predictions, targets)

        # logits are not yet put into a sigmoid layer, because the loss module does this combined
        return logits

    def training_step(self, batch, batch_idx):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # collapse support and query set and train on whole
        support_graphs, query_graphs, support_targets, query_targets = batch
        sub_graphs = support_graphs + query_graphs
        targets = torch.cat([support_targets, query_targets])

        logits = self.forward(sub_graphs, targets, mode='train')

        # logits should be batch size x 1, not batch size x 2!
        # x 2 --> multiple label classification (only if labels are exclusive, can be only one and not multiple)

        loss = self.loss_module(logits, func.one_hot(targets).float())

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"train/loss", loss)

        # back propagate every step, but only log every epoch
        # sum the loss over steps and average at the end of one epoch and then log
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx, dataloader_idx):

        support_graphs, query_graphs, support_targets, query_targets = batch

        if dataloader_idx == 1:
            # Evaluate on meta test set

            # only val query
            # sub_graphs = query_graphs
            # targets = query_targets

            # whole val set
            sub_graphs = support_graphs + query_graphs
            targets = torch.cat([support_targets, query_targets])

            logits = self.forward(sub_graphs, targets, mode='val')

            # TODO: loss has still weights of training balanced set
            loss = self.loss_module(logits, func.one_hot(targets).float())

            # only log this once in the end of an epoch (averaged over steps)
            self.log_on_epoch(f"val/loss", loss)

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default, logs it per epoch (weighted average over batches)

        # only validate on the query set to keep comparability with metamodels
        support_graphs, query_graphs, support_targets, query_targets = batch
        sub_graphs = query_graphs
        targets = query_targets

        self.forward(sub_graphs, targets, mode='test')

# def load_pretrained_encoder(checkpoint_path):
#     """
#     Load a pretrained encoder state dict and remove 'model.' from the keys in the state dict, so that solely
#     the encoder can be loaded.
#
#     Args:
#         checkpoint_path (str) - Path to a checkpoint for the DocumentClassifier.
#     Returns:
#         encoder_state_dict (dict) - Containing all keys for weights of encoder.
#     """
#     checkpoint = torch.load(checkpoint_path)
#     encoder_state_dict = {}
#     for layer, param in checkpoint["state_dict"].items():
#         if layer.startswith("model"):
#             new_layer = layer[layer.index(".") + 1:]
#             encoder_state_dict[new_layer] = param
#
#     return encoder_state_dict


# # noinspection PyProtectedMember
# class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
#     """
#     Learning rate scheduler, combining warm-up with a cosine-shaped learning rate decay.
#     """
#
#     def __init__(self, optimizer, warmup, max_iters):
#         self.warmup = warmup
#         self.max_num_iters = max_iters
#         super().__init__(optimizer)
#
#     def get_lr(self):
#         lr_factor = self.get_lr_factor()
#         return [base_lr * lr_factor for base_lr in self.base_lrs]
#
#     def get_lr_factor(self):
#         current_step = self.last_epoch
#         lr_factor = 0.5 * (1 + np.cos(np.pi * current_step / self.max_num_iters))
#         if current_step < self.warmup:
#             lr_factor *= current_step * 1.0 / self.warmup
#         return lr_factor
