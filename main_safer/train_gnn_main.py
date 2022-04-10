import datetime
import time
import warnings

import torch.nn.functional as func
import wandb

from models.gnn_model import GraphNet
from models.train_utils import get_subgraph_batch

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report
import sys

sys.path.append("..")

import torch.nn as nn

from utils.utils import evaluation_measures, print_stats
from caching_funcs.cache_gnn import *

from optimizers.radam import RiemannianAdam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_summary(predictions, labels):
    target_names = ['False', 'True', 'Unverified']
    print(classification_report(labels, predictions, target_names=target_names))
    return None


class GraphNetMain:
    def __init__(self, config, loaders, vocab_size):
        self.loss_func = config['loss_func']
        self.patience_metric = config['patience_metric']
        self.patience = config['patience']
        self.max_epoch = config['max_epoch']
        self.data_name = config['data_name']

        self.model = GraphNet(config).to(device)
        self.cache_embeds = CacheGNNEmbeds(config, self.model)

        self.vocab_size = vocab_size

        self.train_loader, self.train_val_loader, self.test_loader, self.test_val_loader = loaders

        self.best_val_acc, self.best_val_f1, self.best_val_recall, self.best_val_precision = 0, 0, 0, 0
        self.actual_best_f1 = 0
        self.predictions_list, self.labels_list = [], []
        self.train_losses = []

        self.threshold = 0
        self.prev_val_loss, self.not_improved = 0, 0
        self.best_val_loss = 1e4
        self.terminate_training = False

        self.total_iters = 0

        # self.model_file = os.path.join(self.config['model_checkpoint_path'], self.config['data_name'],
        #                                self.config['model_name'], self.config['model_save_name'])
        # self.config['model_file'] = self.model_file

        self.start_epoch = 1
        self.model_name = config['model_name']

        self.start = time.time()

        if config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'],
                                               weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'RAdam':
            self.optimizer = RiemannianAdam(self.model.parameters(), lr=config['lr'],
                                            weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'],
                                             momentum=config['momentum'], weight_decay=config['weight_decay'])

        if self.loss_func == 'bce_logits':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['pos_wt']]).to(device))
        else:
            self.criterion = nn.BCELoss()

        if config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay_step'],
                                                             gamma=config['lr_decay_factor'])
        elif config['scheduler'] == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[5, 10, 15, 20, 30, 40, 55],
                                                                  gamma=config['lr_decay_factor'])

    def eval_gcn(self):
        self.model.eval()
        predictions_list, labels_list = [], []
        eval_loss = []

        with torch.no_grad():
            for iteration, eval_data in enumerate(self.train_val_loader):

                support_graphs, query_graphs, support_targets, query_targets = eval_data
                sub_graphs = query_graphs
                targets = query_targets

                x, edge_index, cl_mask = get_subgraph_batch(sub_graphs)

                # data_x = data.x * data.representation_mask.unsqueeze(1)

                predictions, node_drop_mask = self.model(x.to(device), edge_index.to(device))
                predictions = predictions[cl_mask]

                if self.loss_func == 'bce':
                    predictions = func.sigmoid(predictions)
                elif self.loss_func == 'bce_logits':
                    predictions = predictions.squeeze(1)

                loss = self.criterion(predictions.to(device), func.one_hot(targets).float().to(device))

                eval_loss.append(loss.detach().item())

                predictions_list.append(self.get_predictions(predictions))
                labels_list.append(targets.cpu().detach().numpy())

            predictions_list = [pred for batch_pred in predictions_list for pred in batch_pred]
            labels_list = [label for batch_labels in labels_list for label in batch_labels]
            eval_loss = sum(eval_loss) / len(eval_loss)

            eval_f1, eval_macro_f1 = evaluation_measures(np.array(predictions_list), np.array(labels_list))

        return eval_f1, eval_macro_f1, eval_loss

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'best_val_f1': self.best_val_f1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.model_file))

    def check_early_stopping(self, eval_f1, eval_macro_f1, eval_loss, epoch):

        metric = eval_f1 if self.patience_metric == 'f1' else eval_loss
        current_best = self.best_val_f1 if self.patience_metric == 'f1' else self.best_val_loss

        new_best = metric > current_best if self.patience_metric == 'f1' else metric < current_best
        if new_best:
            print("New High Score! Saving model...")
            self.best_val_f1 = eval_f1
            self.best_val_loss = eval_loss

            # TODO
            # self.save_model()

        self.scheduler.step()

        # Stopping Criteria based on patience
        diff = metric - current_best if self.patience_metric == 'f1' else current_best - metric
        if diff < 1e-3:
            self.not_improved += 1
            if self.not_improved >= self.patience:
                self.terminate_training = True
        else:
            self.not_improved = 0
        print("current patience: ", self.not_improved)

    def train_epoch_end(self, iterations, epoch):
        self.model.train()

        lr = self.scheduler.get_lr()

        self.total_iters += iterations

        self.predictions_list = [pred for batch_pred in self.predictions_list for pred in batch_pred]
        self.labels_list = [label for batch_labels in self.labels_list for label in batch_labels]

        train_f1, train_macro_f1 = evaluation_measures(np.array(self.predictions_list), np.array(self.labels_list))

        log_wandb(epoch, self.train_losses, train_f1, train_macro_f1)

        # Evaluate on dev set
        eval_f1, eval_macro_f1, eval_loss = self.eval_gcn()

        # print stats
        print_stats(self.max_epoch, epoch, self.train_losses, train_f1, train_macro_f1, eval_loss, eval_f1,
                    eval_macro_f1, self.start, lr[0])

        log_wandb(epoch, eval_loss, eval_f1, eval_macro_f1, val=True)

        # Check for early stopping criteria
        self.check_early_stopping(eval_f1, eval_macro_f1, eval_loss, epoch)

        self.scheduler.step()

        self.predictions_list = []
        self.labels_list = []

    def train_iteration_end(self, predictions, targets):

        if self.loss_func == 'bce':
            predictions = func.sigmoid(predictions)
        elif self.loss_func == 'bce_logits':
            predictions = predictions.squeeze(1)

        loss = self.criterion(predictions.to(device), func.one_hot(targets).float().to(device))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        self.predictions_list.append(self.get_predictions(predictions))
        self.labels_list.append(targets.cpu().detach().numpy())

        self.train_losses.append(loss.detach().item())

    def get_predictions(self, predictions):
        if self.loss_func == 'ce':
            predictions = torch.argmax(func.softmax(predictions, dim=1), dim=1)
        else:
            predictions = func.sigmoid(predictions) if self.loss_func == 'bce_logits' else predictions
            predictions = (predictions > 0.5).type(torch.FloatTensor).argmax(-1)
        return predictions.cpu().detach().numpy()

    def train_main(self, cache=False):
        print("\n\n" + "=" * 100 + "\n\t\t\t\t\t Training Network\n" + "=" * 100)

        self.start = time.time()
        print("\nBeginning training at:  {} \n".format(datetime.datetime.now()))

        for epoch in range(self.start_epoch, self.max_epoch + 1):
            for iteration, data in enumerate(self.train_loader):
                self.model.train()

                support_graphs, query_graphs, support_targets, query_targets = data
                sub_graphs = support_graphs + query_graphs
                targets = torch.cat([support_targets, query_targets])
                x, edge_index, cl_mask = get_subgraph_batch(sub_graphs)

                # data_x = data.x * data.representation_mask.unsqueeze(1)

                predictions, node_drop_mask = self.model(x.to(device), edge_index.to(device))
                predictions = predictions[cl_mask]

                self.train_iteration_end(predictions, targets)

            self.train_epoch_end(iteration, epoch)

            if self.terminate_training:
                break

        # Termination message
        if self.terminate_training:
            print(f"\n{'-' * 100}\nTraining terminated early because the Validation loss did not improve for "
                  f"{self.patience} epochs")
        else:
            print(f"\n{'-' * 100}\nMaximum epochs reached. Finished training !!")

        if cache:
            self.cache_embeds.predict_and_cache(self.data_name)

        # print("\nModel explainer working...")
        # checkpoint = torch.load(self.model_file, map_location=torch.device('cpu'))
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # explainer = GNNExplainer(self.model, epochs=200)
        # node_idx = 20
        # node_feat_mask, edge_mask = explainer.explain_node(node_idx, self.config['data'].x, self.config['data'].edge_index)
        # ax, G = explainer.visualize_subgraph(node_idx, self.config['data'].edge_index, edge_mask, y= self.config['data'].node_type, threshold = None)
        # # plt.savefig(fname = './data/{}_explained_node{}.pdf'.format(self.config['model_name'], str(node_idx)), dpi=25)
        # plt.tight_layout()
        # plt.show()

        return self.best_val_f1


def log_wandb(epoch, loss, f1, lr=0, val=False):
    if not val:
        wandb.log({'train/loss': sum(loss) / len(loss)}, step=epoch)
        wandb.log({'train/f1': f1, 'train/learning_rate': lr}, step=epoch)
    else:
        wandb.log({'val/loss': loss, 'val/f1': f1}, step=epoch)

        # elif not val and config['data_name'] == 'pheme':
        #     f1_micro, f1_macro, f1_weighted = f1
        #     recall_micro, recall_macro, recall_weighted = recall
        #     precision_micro, precision_macro, precision_weighted = prec
        #
        #     writer.add_scalar('train_f1/macro', f1_macro, epoch)
        #     writer.add_scalar('train_f1/micro', f1_micro, epoch)
        #     writer.add_scalar('train_f1/weighted', f1_weighted, epoch)
        #
        #     writer.add_scalar("train/learning_rate", lr, epoch)
