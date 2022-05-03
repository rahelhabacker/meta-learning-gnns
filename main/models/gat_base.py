import torch.nn.functional
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
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
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace + saves config in wandb
        self.save_hyperparameters()

        self.model = GatNet(model_params)

        # flipping the weights
        pos_weight = 1 // model_params["class_weight"][1]
        print(f"Using positive weight: {pos_weight}")
        self.loss_module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.validation_loss = nn.BCEWithLogitsLoss()

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
        predictions = (logits.sigmoid() > 0.5).float()

        for mode_dict, _ in self.metrics.values():
            # shapes should be: pred (batch_size), targets: (batch_size)
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

        # Loss function is not weighted differently

        # 1. valdiation, not balanced --> loss weighting, see what and if it changes; (no loss weighting during training)
        # - keep using full validation set: 1 with balanced, 1 with unbalanced

        # BCE loss
        # BCE with logits loss
        # BCE with Sigmoid and 1 output of the model

        # 1. Multi label loss fixing
        # 2. Loss weighting
        # 3. Sanity Check with train and val on the same split

        loss = self.loss_module(logits, targets.float())

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"train/loss", loss)

        # back propagate every step, but only log every epoch
        # sum the loss over steps and average at the end of one epoch and then log
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx, dataloader_idx):

        support_graphs, query_graphs, support_targets, query_targets = batch

        if dataloader_idx == 1:
            # Evaluate on meta test set

            # testing on a query set that is oversampled should not be happening --> use original distribution
            # training is using a weighted loss --> validation set should use weighted loss as well

            # only val query
            sub_graphs = query_graphs
            targets = query_targets

            # whole val set
            # sub_graphs = support_graphs + query_graphs
            # targets = torch.cat([support_targets, query_targets])

            logits = self.forward(sub_graphs, targets, mode='val')

            # TODO: loss has still weights of training balanced set
            # loss = self.loss_module(logits, func.one_hot(targets).float())
            loss = self.loss_module(logits, targets.float())

            # only log this once in the end of an epoch (averaged over steps)
            self.log_on_epoch(f"val/loss", loss)

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default, logs it per epoch (weighted average over batches)

        # only validate on the query set to keep comparability with metamodels
        support_graphs, query_graphs, support_targets, query_targets = batch
        sub_graphs = query_graphs
        targets = query_targets

        self.forward(sub_graphs, targets, mode='test')
