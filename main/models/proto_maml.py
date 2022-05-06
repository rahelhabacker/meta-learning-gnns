import time
from copy import deepcopy
from statistics import mean, stdev

import torch.nn.functional as func
from torch import optim
from torchmetrics import F1
from tqdm.auto import tqdm

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
from models.proto_net import ProtoNet
from models.train_utils import *
from samplers.batch_sampler import split_list


# noinspection PyAbstractClass
class ProtoMAML(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams):
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

        self.target_class_idx = 1

        pos_weight = 1 // model_params["class_weight"]['train'][self.target_class_idx]
        print(f"Using positive weight: {pos_weight}")
        self.loss_module = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model = GatNet(model_params)

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    def adapt_few_shot(self, x, edge_index, cl_mask, support_targets, mode):

        # Determine prototype initialization
        support_feats = self.model(x, edge_index, mode).squeeze()[cl_mask]

        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)

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
                                self.target_class_idx, self.loss_module)

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

    def outer_loop(self, batch, mode):
        losses = []

        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for graphs, targets in batch:

            support_graphs, query_graphs = split_list(graphs)
            support_targets, query_targets = split_list(targets)

            # Perform inner loop adaptation
            local_model, output_weight, output_bias = self.adapt_few_shot(*get_subgraph_batch(support_graphs),
                                                                          support_targets, mode)

            # Determine loss of query set
            loss, logits = run_model(local_model, output_weight, output_bias, *get_subgraph_batch(query_graphs),
                                     query_targets, mode, self.target_class_idx, self.loss_module)

            self.update_metrics(mode, logits, query_targets)

            # Calculate gradients for query set loss
            if mode == "train":
                # initializes the grads in the outer model, as we used its support features for prototype computation
                # loss.backward()
                self.manual_backward(loss)

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
        self.outer_loop(batch, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)


def run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask, targets, mode, target_class_idx,
              loss_module=None):
    """
    Execute a model with given output layer weights and inputs.
    """

    logits = local_model(x, edge_index, mode)[cl_mask]

    # Expected
    # logits shape batch size x 64
    # output_weight shape 1 x 64
    # output_bias shape 1

    all_class_logits = func.linear(logits, output_weight, output_bias)      # should output: batch size x 1
    logits_target_class = all_class_logits[:, target_class_idx]

    # targets = targets.view(-1, 1) if not len(targets.shape) == 2 else targets
    loss = loss_module(logits_target_class, targets.float()) if loss_module is not None else None

    return loss, logits_target_class


def test_protomaml(model, test_loader, num_classes=1):
    mode = 'test'
    model = model.to(DEVICE)
    model.eval()

    # TODO: use inner loop updates of 200 --> should be higher than in training

    test_start = time.time()

    # Iterate through the full dataset in two manners:
    # First, to select the k-shot batch. Second, to evaluate the model on all other batches.

    f1_fakes = []
    target_class_idx = 1

    for support_batch_idx, batch in tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing"):
        support_graphs, _, support_targets, _ = batch

        # graphs are automatically put to device in adapt few shot
        support_targets = support_targets.to(DEVICE)

        # Finetune new model on support set
        local_model, output_weight, output_bias = model.adapt_few_shot(*get_subgraph_batch(support_graphs),
                                                                       support_targets, mode)

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

                _, pred = run_model(local_model, output_weight, output_bias, *get_subgraph_batch(graphs), targets,
                                    target_class_idx, mode)

                f1_target.update(pred, targets)

            f1_fakes.append(f1_target.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    return (mean(f1_fakes), stdev(f1_fakes)), test_elapsed
