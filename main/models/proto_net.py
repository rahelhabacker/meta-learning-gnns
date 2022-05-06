import time
from statistics import mean, stdev

import numpy as np
import torch.nn
import torchmetrics as tm
from torch import optim
from tqdm.auto import tqdm

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
from models.train_utils import *
from samplers.graph_sampler import KHopSamplerSimple


class ProtoNet(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of Adam optimizer
        """
        super().__init__(validation_sets=['val'])
        self.save_hyperparameters()

        # self.num_classes = model_params["class_weight"].shape[0]

        self.target_class_idx = 1

        # flipping the weights
        pos_weight = 1 // model_params["class_weight"]['train'][self.target_class_idx]
        print(f"Using positive weight: {pos_weight}")
        self.loss_module = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model = GatNet(model_params)

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    @staticmethod
    def calculate_prototypes(features, targets):
        """
        Given a stack of features vectors and labels, return class prototypes.
        :param features:
        :param targets:
        :return:
        """

        # features shape: [N, proto_dim], targets shape: [N]

        # Determine which classes we have
        classes, _ = torch.unique(targets).sort()
        prototypes = []

        for c in classes:
            # noinspection PyTypeChecker
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)

        prototypes = torch.stack(prototypes, dim=0)

        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    @staticmethod
    def classify_features(prototypes, feats):
        """
                Classify new examples with prototypes and return classification error.
                :param prototypes:
                :param feats:
                :param targets:
                :return:
                """
        # Squared euclidean distance
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)

        # predictions = func.log_softmax(-dist, dim=1)      # for CE loss
        # predictions = torch.sigmoid(-dist)  # for BCE loss

        # predictions = -dist  # for BCE with logits loss
        # for BCE with logits loss: don't use negative dist
        logits = dist

        return logits

    def calculate_loss(self, batch, mode):
        """
        Determine training loss for a given support and query set.
        :param batch:
        :param mode:
        :return:
        """

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        support_graphs, query_graphs, support_targets, query_targets = batch

        x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
        support_logits = self.model(x, edge_index, mode)[cl_mask]

        assert support_logits.shape[0] == support_targets.shape[0], \
            "Nr of features returned does not equal nr. of classification nodes!"

        # support logits: episode size x 2, support targets: episode size x 1
        prototypes, classes = ProtoNet.calculate_prototypes(support_logits, support_targets)

        x, edge_index, cl_mask = get_subgraph_batch(query_graphs)
        query_logits = self.model(x, edge_index, mode)[cl_mask]

        assert query_logits.shape[0] == query_targets.shape[0], \
            "Nr of features returned does not equal nr. of classification nodes!"

        logits = ProtoNet.classify_features(prototypes, query_logits)
        # logits and targets: batch size x 1

        # targets have dimensions according to classes which are in the subgraph batch, i.e. if all sub graphs have the
        # same label, targets has 2nd dimension = 1

        logits_target_class = get_fake_logits(logits, self.target_class_idx)

        loss = self.loss_module(logits_target_class, query_targets.float())

        if mode == 'train' or mode == 'val':
            self.log_on_epoch(f"{mode}/loss", loss)

        self.update_metrics(mode, logits_target_class, query_targets)

        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, mode="val")


def get_fake_logits(logits, target_class_idx):
    if logits.shape[1] == 2:
        logits_target_class = logits[:, target_class_idx]
    else:
        # TODO: fix this somehow... now we are using 'real' logits for optimization; maybe project on one dim?
        logits_target_class = logits.squeeze()

    return logits_target_class


@torch.no_grad()
def test_proto_net(model, dataset, data_feats=None, k_shot=4, num_classes=1):
    """
    Use the trained ProtoNet & adapt to test classes. Pick k examples/sub graphs per class from which prototypes are
    determined. Test the metrics on all other sub graphs, i.e. use k sub graphs per class as support set and
    the rest of the dataset as a query set. Iterate through the dataset such that each sub graph has been once
    included in a support set.

     The average performance across all support sets tells how well ProtoNet is expected to perform
     when seeing only k examples per class.

    Inputs
        model - Pretrained ProtoNet model
        dataset - The dataset on which the test should be performed. Should be an instance of TorchGeomGraphDataset.
        data_feats - The encoded features of all articles in the dataset. If None, they will be newly calculated,
                    and returned for later usage.
        k_shot - Number of examples per class in the support set.
    """

    test_node_indices = torch.where(dataset.split_masks['test_mask'])[0]
    sampler = KHopSamplerSimple(dataset, 2)

    model = model.to(DEVICE)
    model.eval()

    # The encoder network remains unchanged across k-shot settings.
    # Hence, we only need to extract the features for all articles once.
    if data_feats is None:

        data_list_collated = []
        for orig_node_idx in test_node_indices:
            data, target = sampler[orig_node_idx]
            data_list_collated.append((data, target))

        sup_graphs, labels = list(map(list, zip(*data_list_collated)))
        x, edge_index, cl_mask = get_subgraph_batch(sup_graphs)
        x, edge_index = x.to(DEVICE), edge_index.to(DEVICE)
        feats = model.model(x, edge_index, 'test')
        feats = feats[cl_mask]

        node_features = feats.detach().cpu()  # shape: 1975 x 2
        node_targets = torch.tensor(labels)  # shape: 1975

        node_targets, sort_idx = node_targets.sort()
        node_features = node_features[sort_idx]
    else:
        node_features, node_targets = data_feats

    # Iterate through the full dataset in two manners:
    #   First, to select the k-shot batch.
    #   Second, to evaluate the model on all the other examples

    test_start = time.time()

    start_indices_per_class = torch.tensor((num_classes, 1))
    for c in range(num_classes):
        start_indices_per_class[c] = torch.where(node_targets == c)[0][0].item()

    accuracies, f1_fake, f1_macros = [], [], []
    for k_idx in tqdm(range(0, node_features.shape[0], k_shot), "Evaluating prototype classification", leave=False):
        # Select support set (k examples per class) and calculate prototypes
        k_node_feats, k_targets = get_as_set(k_idx, k_shot, node_features, node_targets, start_indices_per_class)
        prototypes, classes = model.calculate_prototypes(k_node_feats, k_targets)

        batch_f1_target = tm.F1(num_classes=num_classes, average='none', multiclass=False)

        for e_idx in range(0, node_features.shape[0], k_shot):
            if k_idx == e_idx:  # Do not evaluate on the support set examples
                continue

            e_node_feats, e_targets = get_as_set(e_idx, k_shot, node_features, node_targets, start_indices_per_class)
            logits = model.classify_features(prototypes, e_node_feats)

            logits_target_class = get_fake_logits(logits, 1)
            predictions = (logits_target_class.sigmoid() > 0.5).long().squeeze()
            batch_f1_target.update(predictions, e_targets)

        # F1 values can be nan, if e.g. proto_classes contains only one of the 2 classes
        f1_fake_value = batch_f1_target.compute().item()
        if not np.isnan(f1_fake_value):
            f1_fake.append(f1_fake_value)

        batch_f1_target.reset()

    test_end = time.time()
    test_elapsed = test_end - test_start

    return (mean(f1_fake), stdev(f1_fake)), test_elapsed, (node_features, node_targets)


def get_as_set(idx, k_shot, all_node_features, all_node_targets, start_indices_per_class):
    node_targets = []
    node_feats = []
    for s_idx in start_indices_per_class:
        start_idx = idx + s_idx
        targets = all_node_targets[start_idx: start_idx + k_shot]
        feats = all_node_features[start_idx: start_idx + k_shot]
        node_targets.append(targets)
        node_feats.append(feats)
    return torch.cat(node_feats), torch.cat(node_targets)
