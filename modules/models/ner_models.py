from modules.layers.decoders import *
from modules.layers.embedders import *
from modules.layers.layers import BiLSTM, MultiHeadAttention
import abc

class BERTNerModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all BERT Models"""

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError("abstract method forward must be implemented")

    @abc.abstractmethod
    def score(self, batch):
        raise NotImplementedError("abstract method score must be implemented")

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError("abstract method create must be implemented")

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp

class BERTBiLSTMNCRF(BERTNerModel):

    def __init__(self, embeddings, lstm, crf, device="cuda"):
        super(BERTBiLSTMNCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.forward(output, labels_mask)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
                embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = NCRFDecoder.create(
            label_size, hidden_dim, crf_dropout, nbest, device=device)
        return cls(embeddings, lstm, crf, device)

class BERTAttnNCRF(BERTNerModel):

    def __init__(self, embeddings, attn, crf, device="cuda"):
        super(BERTAttnNCRF, self).__init__()
        self.embeddings = embeddings
        self.attn = attn
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.forward(output, labels_mask)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        attn = MultiHeadAttention(key_dim, val_dim, embedding_size, num_heads, attn_dropout)
        crf = NCRFDecoder.create(
            label_size, embedding_size, crf_dropout, nbest=nbest, device=device)
        return cls(embeddings, attn, crf, device)

class BERTBiLSTMAttnNCRF(BERTNerModel):

    def __init__(self, embeddings, lstm, attn, crf, device="cuda"):
        super(BERTBiLSTMAttnNCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.attn = attn
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.forward(output, labels_mask)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = NCRFDecoder.create(
            label_size, hidden_dim, crf_dropout, nbest=nbest, device=device)
        return cls(embeddings, lstm, attn, crf, device)


class RoBERTBiLSTMAttnNCRF(BERTNerModel):

    def __init__(self, embeddings, lstm, attn, crf, device="cuda"):
        super(RoBERTBiLSTMAttnNCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.attn = attn
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.forward(output, labels_mask)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = RoBERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = NCRFDecoder.create(
            label_size, hidden_dim, crf_dropout, nbest=nbest, device=device)
        return cls(embeddings, lstm, attn, crf, device)