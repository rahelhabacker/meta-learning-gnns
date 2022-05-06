import time

from torch import nn

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
from models.train_utils import *


# noinspection PyAbstractClass
class GatBase(GraphTrainer):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, val_batches):
        """
        Args:
            model_params - Hyperparameters for the whole model, as dictionary.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__(validation_sets=['val_support', 'val_query'])

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace + saves config in wandb
        self.save_hyperparameters()

        self.model = GatNet(model_params)

        train_weight = 1 // model_params["class_weight"]['train'][1]
        print(f"Using positive train weight: {train_weight}")
        self.loss_module = nn.BCEWithLogitsLoss(pos_weight=train_weight)

        val_weight = 1 // model_params["class_weight"]['val'][1]
        print(f"Using positive validation weight: {val_weight}")
        self.validation_loss = nn.BCEWithLogitsLoss(pos_weight=val_weight)

        # Deep copy of the model: one for train, one for val --> update validation model with weights from train model
        # validation fine-tuning should happen on a copy of the model NOT on the model which is trained
        # --> Training should not be affected by validation
        self.validation_model = GatNet(model_params)

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer_hparams

        train_optimizer, train_scheduler = self.get_optimizer(opt_params['lr'], opt_params['lr_decay_epochs'])
        val_optimizer, val_scheduler = self.get_optimizer(opt_params['lr_val'], opt_params['lr_decay_epochs_val'],
                                                          self.validation_model)
        optimizers = [train_optimizer, val_optimizer]

        schedulers = []
        if train_scheduler is not None:
            schedulers.append(train_scheduler)
        if val_scheduler is not None:
            schedulers.append(val_scheduler)

        return optimizers, schedulers

    def forward(self, graphs, targets, mode=None):

        # make a batch out of all sub graphs and push the batch through the model
        x, edge_index, cl_mask = get_subgraph_batch(graphs)

        logits = self.model(x, edge_index, mode)[cl_mask].squeeze()

        self.update_metrics(mode, logits, targets)

        # logits are not yet put into a sigmoid layer, because the loss module does this combined
        return logits

    def training_step(self, batch, batch_idx):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_opt, _ = self.optimizers()

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

        train_opt.zero_grad()
        self.manual_backward(loss)
        train_opt.step()

        # TODO: Accumulate gradients?
        # self.manual_backward(loss)
        # n = 10
        # accumulate gradients of N batches
        # if (batch_idx + 1) % n == 0:
        #     train_opt.step()
        #     train_opt.zero_grad()

        # step every N epochs
        train_scheduler, _ = self.lr_schedulers()
        # print(f"Train SD, step size: {train_scheduler.step_size}")
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % train_scheduler.step_size == 0:
            # print(f"Trainer epoch: {self.trainer.current_epoch + 1}")
            # print("Reducing Train LR")
            # print(f"LR before: {train_scheduler.get_last_lr()}")
            train_scheduler.step()
            # print(f"LR after: {train_scheduler.get_last_lr()}")

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"train/loss", loss)

        # back propagate every step, but only log every epoch
        # sum the loss over steps and average at the end of one epoch and then log
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx, dataloader_idx):

        # update the weights of the validation model with weights from trained model
        self.validation_model.load_state_dict(self.model.state_dict())

        support_graphs, query_graphs, support_targets, query_targets = batch

        if dataloader_idx == 0:
            mode = 'val_support'

            # Validation requires to finetune a model, hence we need to enable gradients
            torch.set_grad_enabled(True)

            self.validation_model.train()

            _, val_optimizer = self.optimizers()

            x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
            logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()

            self.update_metrics(mode, logits, support_targets)

            loss = self.validation_loss(logits, support_targets.float())

            self.log_on_epoch(f"{mode}/loss", loss)

            # Calculate gradients and perform finetune update
            val_optimizer.zero_grad()
            self.manual_backward(loss)
            val_optimizer.step()

            # step every N epochs
            _, val_scheduler = self.lr_schedulers()
            # print(f"Val SD, step size: {val_scheduler.step_size}")
            # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % val_scheduler.step_size == 0:
            #     print(f"Trainer epoch: {self.trainer.current_epoch + 1}")
            #     print("Reducing Val LR")
            # val_scheduler.step()

            # print(f"Val batch_idx: {batch_idx}")
            if self.hparams['val_batches'] == (batch_idx + 1):
                # print(f"Trainer epoch: {self.trainer.current_epoch + 1}")
                # print("Reducing Val LR")
                # print(f"Val LR before: {val_scheduler.get_last_lr()}")
                val_scheduler.step()
                # print(f"Val LR after: {val_scheduler.get_last_lr()}")

            # SGD does not keep any state --> Create an SGD optimizer again every time
            # I enter the validation epoch; global or local should not be a difference
            # different for ADAM --> Keeps running weight parameter, that changes
            # per epoch, keeps momentum

            # Main constraint: Use same optimizer as in training, global ADAM validation

            torch.set_grad_enabled(False)

        elif dataloader_idx == 1:
            # Evaluate on meta test set
            mode = 'val_query'

            # testing on a query set that is oversampled should not be happening --> use original distribution
            # training is using a weighted loss --> validation set should use weighted loss as well
            # with extra validation model
            x, edge_index, cl_mask = get_subgraph_batch(query_graphs)
            logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()
            loss = self.validation_loss(logits, query_targets.float())

            self.update_metrics(mode, logits, query_targets)

            # only log this once in the end of an epoch (averaged over steps)
            self.log_on_epoch(f"{mode}/loss", loss)

    def test_step(self, batch, batch_idx1, batch_idx2):
        """
        By default, logs it per epoch (weighted average over batches). Only validates on the query set of each batch to
        keep comparability with the meta learners.
        """

        _, sub_graphs, _, targets = batch
        self.forward(sub_graphs, targets, mode='test')


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
    val_f1_fake = val_results['test/f1_fake']

    test_end = time.time()
    elapsed = test_end - test_start

    return test_f1_fake, val_f1_fake, elapsed
