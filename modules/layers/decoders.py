import torch
from torch.nn import functional
from torch.autograd import Variable
from torch import nn
from layers import Linears
from ncrf import NCRF

class NCRFDecoder(nn.Module):

    def __init__(self,
                 crf, label_size, input_dim, input_dropout=0.5, nbest=8):
        super(NCRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.nbest = nbest
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        inputs = self.dropout(inputs)

        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, labels_mask):
        logits = self.forward_model(inputs)
        _, preds = self.crf._viterbi_decode_nbest(logits, labels_mask, self.nbest)
        preds = preds[:, :, 0]
        return preds

    def score(self, inputs, labels_mask, labels):
        logits = self.forward_model(inputs)
        crf_score = self.crf.neg_log_likelihood_loss(logits, labels_mask, labels) / logits.size(0)
        return crf_score

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5, nbest=8, device="cuda"):
        return cls(NCRF(label_size, device), label_size + 2, input_dim, input_dropout, nbest)
