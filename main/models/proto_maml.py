import time
from copy import deepcopy
from statistics import mean, stdev

import torch.nn
import torch.nn.functional as func
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1
from tqdm.auto import tqdm

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer, get_or_none
from models.proto_net import ProtoNet
from models.train_utils import *
from samplers.episode_sampler import split_list


# noinspection PyAbstractClass
class ProtoMAML(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, other_params):
        """
        Inputs
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            n_inner_updates - Number of inner loop updates to perform
        """
        super().__init__(validation_sets=['val'])
        self.save_hyperparameters()

        self.n_inner_updates = model_params['n_inner_updates']
        self.n_inner_updates_test = model_params['n_inner_updates_test']

        self.lr_inner = self.hparams.optimizer_hparams['lr_inner']

        self.k_shot_support = other_params['k_shot_support']

        train_weight = get_or_none(other_params, 'train_loss_weight')
        print(f"Positive train weight: {train_weight}")
        self.train_loss_module = BCEWithLogitsLoss(pos_weight=train_weight)

        val_weight = get_or_none(other_params, 'val_loss_weight')
        print(f"Positive val weight: {val_weight}")
        self.val_loss_module = BCEWithLogitsLoss(pos_weight=val_weight)

        self.model = GatNet(model_params)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    def adapt_few_shot(self, x, edge_index, cl_mask, support_targets, mode, loss_module):

        # Determine prototype initialization
        support_feats = self.model(x, edge_index, mode).squeeze()[cl_mask]

        prototypes = ProtoNet.calculate_prototypes(support_feats, support_targets)

        # Copy model for inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.lr_inner)
        local_optim.zero_grad()

        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1) ** 2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        updates = self.n_inner_updates if mode == 'train' else self.n_inner_updates_test

        # Optimize inner loop model on support set
        for _ in range(updates):
            # Determine loss on the support set
            loss, _ = run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask, support_targets, mode,
                                loss_module)

            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()

            # Update output layer via SGD
            output_weight.data -= self.hparams.optimizer_hparams['lr_output'] * output_weight.grad
            output_bias.data -= self.hparams.optimizer_hparams['lr_output'] * output_bias.grad

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias

    def outer_loop(self, batch, loss_module, mode):
        losses = []

        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for graphs, targets in batch:
            # This should be done in the graph sampler
            support_graphs, query_graphs = split_list(graphs, self.k_shot_support * 2)
            support_targets, query_targets = split_list(targets, self.k_shot_support * 2)

            # Perform inner loop adaptation
            local_model, output_weight, output_bias = self.adapt_few_shot(*get_subgraph_batch(support_graphs),
                                                                          support_targets, mode, loss_module)

            # Determine loss of query set
            loss, query_predictions = run_model(local_model, output_weight, output_bias,
                                                *get_subgraph_batch(query_graphs), query_targets, mode, loss_module)

            self.update_metrics(mode, query_predictions, query_targets)

            # Calculate gradients for query set loss
            if mode == "train":
                # initializes the grads in the outer model, as we used its support features for prototype computation
                loss.backward()

                for i, (p_global, p_local) in enumerate(zip(self.model.parameters(), local_model.parameters())):
                    if p_global.requires_grad is False:
                        continue
                    # First-order approx. -> add gradients of fine-tuned and base model
                    p_global.grad += p_local.grad

            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            # noinspection PyUnresolvedReferences
            opt.zero_grad()

        self.log_on_epoch(f"{mode}/loss", sum(losses) / len(losses))

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, self.train_loss_module, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, self.val_loss_module, mode="val")
        torch.set_grad_enabled(False)


def run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask, targets, mode, loss_module=None):
    """
    Execute a model with given output layer weights and inputs.
    """

    logits = local_model(x, edge_index, mode)[cl_mask]

    # Expected
    # logits shape:         batch size x 64
    # output_weight shape:  2 x 64
    # output_bias shape:    2

    # out:                  80 x 2
    logits = func.linear(logits, output_weight, output_bias)

    # now we have logits for both prototype classes --> use only logits for the target class
    logits_fake = logits[:, 1].view(-1, 1)

    targets = targets.view(-1, 1) if not len(targets.shape) == 2 else targets
    loss = loss_module(logits_fake, targets.float()) if loss_module is not None else None

    return loss, (logits_fake.sigmoid() > 0.5).float()


def test_protomaml(model, test_loader, num_classes=1):
    mode = 'test'
    model = model.to(DEVICE)
    model.eval()

    # TODO: use inner loop updates of 200 --> should be higher than in training

    test_start = time.time()

    # Iterate through the full dataset in two manners:
    # First, to select the k-shot batch. Second, to evaluate the model on all other batches.

    test_loss_weight = None
    loss_module = BCEWithLogitsLoss(pos_weight=test_loss_weight)
    f1_fakes = []

    for support_batch_idx, batch in tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing"):
        support_graphs, _, support_targets, _ = batch

        # graphs are automatically put to device in adapt few shot
        support_targets = support_targets.to(DEVICE)

        # Finetune new model on support set
        local_model, output_weight, output_bias = model.adapt_few_shot(*get_subgraph_batch(support_graphs),
                                                                       support_targets, mode, loss_module)

        f1_target = F1(num_classes=num_classes, average='none').to(DEVICE)

        with torch.no_grad():  # No gradients for query set needed
            local_model.eval()

            # Evaluate all examples in test dataset
            for query_batch_idx, test_batch in enumerate(test_loader):

                if support_batch_idx == query_batch_idx:
                    # Exclude support set elements
                    continue

                support_graphs, query_graphs, support_targets, query_targets = batch
                graphs = support_graphs + query_graphs
                targets = torch.cat([support_targets, query_targets]).to(DEVICE)

                _, pred = run_model(local_model, output_weight, output_bias, *get_subgraph_batch(graphs), targets, mode)

                f1_target.update(pred, targets)

            f1_fakes.append(f1_target.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    return (mean(f1_fakes), stdev(f1_fakes)), test_elapsed
