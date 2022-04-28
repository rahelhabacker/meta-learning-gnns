import torch.cuda

from data_prep.graph_dataset import TorchGeomGraphDataset
from data_prep.graph_preprocessor import SPLITS
from samplers.batch_sampler import FewShotSampler, get_n_query_for_samples, BatchSampler
from samplers.graph_sampler import KHopSampler
from samplers.maml_batch_sampler import FewShotMamlSampler

SUPPORTED_DATASETS = ['gossipcop', 'twitterHateSpeech']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data(data_train, data_eval, model_name, hop_size, top_k, top_users_excluded, k_shot, train_split_size,
             eval_split_size, feature_type, vocab_size, dirs, batch_size, num_workers=None, balance_data=False):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_train (str): Name of the data corpus which should be used for training.
        data_eval (str): Name of the data corpus which should be used for testing/evaluation.
        model_name (str): Name of the model should be used.
        hop_size (int): Number of hops used to create sub graphs.
        top_k (int): Number (in thousands) of top users to be used in graph.
        top_users_excluded (int): Number (in percentage) of top users who should be excluded.
        k_shot (int): Number of examples used per task/batch.
        train_split_size (tuple): Floats defining the size of test/train/val for the training dataset.
        eval_split_size (tuple): Floats defining the size of test/train/val for the evaluation dataset.
        feature_type (str): Type of features that should be used.
        vocab_size (int): Size of the vocabulary.
        dirs (tuple): Path to the data (full & complete) to be used to create the graph (feature file, edge file etc.)
        num_workers (int): Amount of workers for parallel processing.
    Raises:
        Exception: if the data_name is not in SUPPORTED_DATASETS.
    """

    if data_train not in SUPPORTED_DATASETS:
        raise ValueError(f"Train data with name '{data_train}' is not supported.")

    if data_eval not in SUPPORTED_DATASETS:
        raise ValueError(f"Eval data with name '{data_eval}' is not supported.")

    if data_train == data_eval:
        assert train_split_size[0] > 0.0 and train_split_size[1] and train_split_size[2] > 0.0, \
            "Data for training and evaluation is equal and one of the split sizes is 0!"

    data_config = {'top_users': top_k, 'top_users_excluded': top_users_excluded, 'feature_type': feature_type,
                   'vocab_size': vocab_size, 'balance_data': balance_data
                   }
    # TODO: only temporary for testing
    # ,'balance_val': True}

    # creating a train and val loader from the train dataset
    train_config = {**data_config, **{'data_set': data_train, 'train_size': train_split_size[0],
                                      'val_size': train_split_size[1], 'test_size': train_split_size[2]}}
    graph_data_train = TorchGeomGraphDataset(train_config, train_split_size, *dirs)

    n_query_train = get_max_n_query(graph_data_train)
    print(f"\nUsing max query samples for episode creation: {n_query_train}\n")

    train_loader = get_loader(graph_data_train, model_name, hop_size, k_shot, num_workers, 'train', n_query_train,
                              batch_size)
    train_val_loader = get_loader(graph_data_train, model_name, hop_size, k_shot, num_workers, 'val', n_query_train,
                                  batch_size)

    print(f"\nTrain graph size: \n num_features: {graph_data_train.size[1]}\n total_nodes: {graph_data_train.size[0]}")

    if data_train == data_eval:
        print(f'\nData eval and data train are equal, loading graph data only once.\n')
        graph_data_eval = graph_data_train
        test_val_loader = train_val_loader
        n_query_eval = n_query_train
    else:
        # creating a val and test loader from the eval dataset
        data_config['top_users_excluded'] = 0

        eval_config = {**data_config, **{'data_set': data_eval, 'train_size': eval_split_size[0],
                                         'val_size': eval_split_size[1], 'test_size': eval_split_size[2]}}

        graph_data_eval = TorchGeomGraphDataset(eval_config, eval_split_size, *dirs)
        n_query_eval = get_max_n_query(graph_data_eval)

        print(f"\nTest graph size: \n num_features: {graph_data_eval.size[1]}\n total_nodes: {graph_data_eval.size[0]}")

        test_val_loader = get_loader(graph_data_eval, model_name, hop_size, k_shot, num_workers, 'val', n_query_eval,
                                     batch_size)

    test_loader = get_loader(graph_data_eval, model_name, hop_size, k_shot, num_workers, 'test', n_query_eval,
                             batch_size)

    # verify_not_overlapping_samples(train_loader)
    # verify_not_overlapping_samples(train_val_loader)
    # verify_not_overlapping_samples(test_val_loader)
    # verify_not_overlapping_samples(test_loader)

    return (train_loader, train_val_loader, test_loader, test_val_loader), graph_data_train, graph_data_eval


def get_num_workers(sampler, num_workers):
    if num_workers is not None:
        return num_workers
    elif not torch.cuda.is_available():
        # mac has 8 CPUs
        return 0
    elif type(sampler) == BatchSampler:
        for r in reversed(range(2, 11)):
            if (len(sampler) / r) >= 2:
                return r
    elif type(sampler) == FewShotSampler:
        return 4
    elif type(sampler) == FewShotMamlSampler:
        return 1
    return 0


# def get_graph_sampler(sampler_type, data):
#     clusters = 5
#     batch_size = 344
#     shuffle = False
#
#     if sampler_type == 'cluster':
#         cluster_data = ClusterData(data, num_parts=clusters, recursive=False)
#         return ClusterLoader(cluster_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
#     elif sampler_type == 'random_walk':
#         return GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2, num_steps=5, sample_coverage=100,
#                                            num_workers=0)
#     elif sampler_type == 'node':
#         return GraphSAINTNodeSampler(data, batch_size=6000, num_steps=5, sample_coverage=100, num_workers=0)
#     elif sampler_type == 'edge':
#         return GraphSAINTEdgeSampler(data, batch_size=6000, num_steps=5, sample_coverage=100, num_workers=0)


def get_loader(graph_data, model_name, hop_size, k_shot, num_workers, mode, n_queries, batch_size):
    n_classes = len(graph_data.labels)

    mask = graph_data.mask(f"{mode}_mask")

    targets = graph_data.data.y[mask]
    max_n_query = n_queries[mode]

    if model_name == 'gat' and mode == 'train':
        batch_sampler = BatchSampler(targets, max_n_query, mode, batch_size, n_way=n_classes, k_shot=k_shot)
    elif model_name == 'prototypical' or (model_name == 'gat' and mode != 'train'):
        batch_sampler = FewShotSampler(targets, max_n_query, mode, n_way=n_classes, k_shot=k_shot)
    elif model_name == 'gmeta':
        batch_size = batch_size if mode == 'train' else 2
        batch_sampler = FewShotMamlSampler(targets, max_n_query, mode, n_classes, k_shot, batch_size)
    else:
        raise ValueError(f"Model with name '{model_name}' is not supported.")

    num_workers = get_num_workers(batch_sampler, num_workers)

    print(f"Num workers: {num_workers}")

    sampler = KHopSampler(graph_data, model_name, batch_sampler, n_classes, k_shot, hop_size, mode,
                          num_workers=num_workers)

    # sampler = get_graph_sampler('random_walk', graph_data)

    print(f"{mode} sampler episodes / batches: {len(sampler)}\n")

    # no need to wrap it again in a dataloader
    return sampler


def get_max_n_query(graph_data):
    """
    Calculates the number of query samples which fits the maximum configured shot number and all other available
    shot numbers. This is required in order to keep the query sets static throughout training with different shot sizes.
    """

    n_classes = len(graph_data.labels)

    n_queries = {}
    for split in SPLITS:
        # maximum amount of query samples which should be used from the total amount of samples
        samples = int(len(torch.where(graph_data.split_masks[f"{split}_mask"])[0]) / n_classes)
        n_queries[split] = get_n_query_for_samples(samples, n_classes)

    return n_queries


def verify_not_overlapping_samples(loader):
    """
    Runs check to verify that support and query set have the same classes but distinct examples.
    """

    n_class1_diff, n_class2_diff, support_duplicates, query_duplicates, num_equals = 0, 0, 0, 0, 0

    for batch in iter(loader):
        if type(loader.b_sampler) != FewShotMamlSampler:
            support_graphs, query_graphs, support_targets, query_targets = batch
        else:
            support_graphs, query_graphs, support_targets, query_targets = [], [], [], []
            for task in batch:
                sub_graphs, targets = task

                for i, graph in enumerate(sub_graphs):
                    if graph.set_type == 'support':
                        support_graphs.append(graph)
                        support_targets.append(targets[i])
                    elif graph.set_type == 'query':
                        query_graphs.append(graph)
                        query_targets.append(targets[i])

        # Support and query should have same number of examples...

        # support_bins = torch.bincount(support_targets)
        # query_bins = torch.bincount(query_targets)
        # difference = torch.abs(support_bins - query_bins)
        #
        # if difference.shape[0] > 0:
        #     n_class1_diff += difference[0]
        #
        # if difference.shape[0] > 1:
        #     n_class2_diff += difference[1]

        # ... but different examples within each set ...
        support_center_idx = [s.orig_center_idx for s in support_graphs]
        query_center_idx = [s.orig_center_idx for s in query_graphs]

        s_duplicates = abs(len(support_center_idx) - len(set(support_center_idx)))
        support_duplicates += s_duplicates

        q_duplicates = abs(len(query_center_idx) - len(set(query_center_idx)))
        query_duplicates += q_duplicates

        # ... and different examples across all sets.
        n_equals = set(support_center_idx).intersection(set(query_center_idx))
        num_equals += len(n_equals)

    # difference in n query and n support
    # assert n_class1_diff == 0, \
    #     f"Class 1: Difference in number of samples in query and support is not 0 but: {n_class1_diff}!"
    # assert n_class2_diff == 0, \
    #     f"Class 2: Difference in number of samples in query and support is not 0 but: {n_class2_diff}!"

    assert support_duplicates == 0, f"Support set contains {support_duplicates} duplicates!"
    assert query_duplicates == 0, f"Query set contains {query_duplicates} duplicates!"

    assert num_equals == 0, f"Query and support set intersect with {num_equals} samples!"
