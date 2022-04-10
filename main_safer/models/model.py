from __future__ import absolute_import, division, print_function

import sys
import warnings

from torch import nn
from torch.nn import Parameter

# from simpletransformers.classification import ClassificationModel
# from simpletransformers.experimental.classification import classification_model

warnings.filterwarnings("ignore")
sys.path.append("..")

from utils.utils import *
from utils.data_utils_gnn import *
from utils.data_utils_txt import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################
## Helper Functions ##
######################

"""
The following 2 implementations are taken from the implementation of LSTM-reg in the HEDWIG framework
(https://github.com/castorini/hedwig/tree/master/models/reg_lstm)

"""


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse)
    return X


class WeightDrop_manual(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def null_function(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.null_function

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                    mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                    w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training).to(device)
            setattr(self.module, name_w, w)

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                    mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                    w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training).to(device)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


###################
## Model Classes ##
###################


class Document_Classifier(nn.Module):
    """
    Main class that controls training and calling of other classes based on corresponding model_name

    """

    def __init__(self, config, pre_trained_embeds=None):
        super(Document_Classifier, self).__init__()

        self.lstm_dim = config['lstm_dim']
        self.model_name = config['model_name']
        self.embed_name = config['embed_dim']
        self.fc_dim = config['fc_dim']
        self.num_classes = config['n_classes']
        self.embed_dim = config['embed_dim'] if config['embed_dim'] == 300 else 2 * config['embed_dim']
        self.batch_size = config['batch_size']
        self.num_kernels = config["kernel_num"]
        self.kernel_sizes = [int(k) for k in config["kernel_sizes"].split(',')]
        self.mode = 'single' if not config['parallel_computing'] else 'multi'

        # Choose the right embedding method based on embed_dim given
        if config['embed_dim'] == 300:
            self.vocab_size = config['vocab_size']
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.embedding.weight.data.copy_(pre_trained_embeds)
            self.embedding.requires_grad = False
        elif config['embed_dim'] == 128:
            # Small
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
            self.elmo = Elmo(options_file=self.options_file, weight_file=self.weight_file, num_output_representations=1,
                             requires_grad=False)
        elif config['embed_dim'] == 256:
            # Medium
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
            self.elmo = Elmo(options_file=self.options_file, weight_file=self.weight_file, num_output_representations=1,
                             requires_grad=False)
        elif config['embed_dim'] == 512:
            # Highest
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            self.elmo = Elmo(options_file=self.options_file, weight_file=self.weight_file, num_output_representations=1,
                             requires_grad=False)

        self.classifier = nn.Sequential(nn.Dropout(config["dropout"]),
                                        nn.Linear(self.fc_inp_dim, self.fc_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.fc_dim, self.num_classes))

    def forward(self, inp, sent_lens=0, doc_lens=0, arg=0, cache=False, attn_mask=None):

        if self.model_name in ['bilstm', 'bilstm_pool', 'cnn']:
            if self.embed_name == 'glove':
                inp = self.embedding(inp)
            else:
                inp = self.elmo(inp.contiguous())['elmo_representations'][0]

            out = self.encoder(inp.contiguous(), lengths=sent_lens)

        elif self.embed_name in ['dbert', 'roberta']:
            out = self.encoder(inp, attn_mask)

        # for HAN the embeddings are taken care of in its model class
        elif self.model_name == 'han':
            if self.embed_dim == 300:
                inp = self.embedding(inp)
                # out = self.encoder(inp, embedding = self.embedding, sent_lengths = sent_lens, doc_lengths = doc_lens)
                out = self.encoder(inp, sent_lengths=sent_lens, num_sent_per_document=doc_lens, arg=arg)
            else:
                if self.mode == 'multi':
                    inp = inp.reshape((inp.shape[0] * inp.shape[1], inp.shape[2], inp.shape[3]))
                    sent_lens = sent_lens.reshape((sent_lens.shape[0] * sent_lens.shape[1]))
                inp = self.elmo(inp.contiguous())['elmo_representations'][0]
                out = self.encoder(inp, sent_lengths=sent_lens, num_sent_per_document=doc_lens, arg=arg)

        if not cache:
            out = self.classifier(out)

        return out
