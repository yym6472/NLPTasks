"""
This python module contains the implementation of (DatasetReader, Model, Predictor)
for Sentiment Polarity Detecting (SPD) task, based on the AllenNLP framework.
"""


import torch
import numpy as np

from typing import Iterator, List, Dict, Tuple
from overrides import overrides

from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, MultiHeadSelfAttention
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import Predictor
from allennlp.common import JsonDict
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.dataset_readers.dataset_utils.span_utils import bmes_tags_to_spans


@TokenIndexer.register("clues")
class CluesTokenIndexer(TokenIndexer[int]):
    """
    Index tokens based on clues dict.
    """
    def __init__(self,
                 subj_clues: Dict[str, Dict[str, str]],
                 namespace: str = 'clues',
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self.subj_clues = subj_clues
        self.namespace = namespace

    def _get_token_label(self, token: Token):
        if token.text not in self.subj_clues:
            return "neutral"
        else:
            return (self.subj_clues[token.text]["type"][0:-4] + "_" +
                   self.subj_clues[token.text]["priorpolarity"][0:3])

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If `text_id` is set on the token (e.g., if we're using some kind of hash-based word
        # encoding), we will not be using the vocab for this token.
        if getattr(token, 'text_id', None) is None:
            label = self._get_token_label(token)
            counter[self.namespace][label] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in tokens:
            if getattr(token, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead.
                indices.append(token.text_id)
            else:
                label = self._get_token_label(token)
                indices.append(vocabulary.get_token_index(label, self.namespace))

        return {index_name: indices}

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}


@DatasetReader.register("spd_dataset_reader")
class SPDDatasetReader(DatasetReader):

    def __init__(self,
                 subj_clues_path: str = None) -> None:
        super().__init__(lazy=False)
        if subj_clues_path:
            self.subj_clues = {}
            with open(subj_clues_path, "r", encoding="utf-8") as f:
                for line in f:
                    # keys: [type, len, word1, pos1, stemmed1, priorpolarity]
                    clue_dict = {item.split("=")[0]: item.split("=")[1]
                                for item in line.strip().split(" ")
                                if "=" in item}
                    if clue_dict["priorpolarity"] not in ("positive", "negative"):
                        continue
                    word = clue_dict["word1"]
                    if word in self.subj_clues:
                        self.subj_clues.pop(word)
                    else:
                        assert clue_dict["type"] in ("strongsubj", "weaksubj")
                        assert clue_dict["priorpolarity"] in ("positive", "negative")
                        self.subj_clues[word] = {
                            "type": clue_dict["type"],
                            "priorpolarity": clue_dict["priorpolarity"]
                        }
        else:
            self.subj_clues = None
    
    def text_to_instance(self,
                         sentence: List[str],
                         label: str = None) -> Instance:
        if self.subj_clues is not None:
            sentence_field = TextField([Token(token) for token in sentence],
                                       {"tokens": SingleIdTokenIndexer(), 
                                        "clues": CluesTokenIndexer(self.subj_clues)})
        else:
            sentence_field = TextField([Token(token) for token in sentence],
                                       {"tokens": SingleIdTokenIndexer()})
        fields = {"sentence": sentence_field}
        if label:  # "pos" or "neg"
            label_field = LabelField(label=label, label_namespace="label")
            fields["label"] = label_field
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                sentence, label = line.strip().split("\t")
                tokens = sentence.strip().split(" ")
                yield self.text_to_instance(tokens, label)


@Model.register("spd_model")
class SPDModel(Model):
    """
    A allennlp model implementation for Sentiment Polarity Detecting (SPD).
    The model implementation follows Embeddings + Self-Attention + Bi-LSTM +
    Linear architecture, see `forward()` method for more details.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 seq2seq_encoder: Seq2SeqEncoder,
                 seq2vec_encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.final2label = torch.nn.Linear(in_features=seq2vec_encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size("label"))
        self.acc = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        seq2seq_encoder_out = self.seq2seq_encoder(embeddings, mask)
        seq2vec_encoder_out = self.seq2vec_encoder(seq2seq_encoder_out, mask)
        logits = self.final2label(seq2vec_encoder_out)
        output = {"logits": logits}
        
        if label is not None:
            self.acc(logits, label)
            output["loss"] = self._compute_loss(logits, label)
        return output
    
    def _compute_loss(self,
                      logits: torch.Tensor,
                      label: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, num_classes)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        label = label.view(-1, 1).long()
        # shape: (batch_size, 1), pylint: disable=no-member
        negative_log_likelihood = - torch.gather(log_probs, dim=1, index=label)
        # shape: (batch_size)
        nagative_log_likelihood = negative_log_likelihood.view(*label.size())
        return nagative_log_likelihood.sum() / nagative_log_likelihood.size()[0]

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.acc.get_metric(reset)
        return {"accuracy": accuracy}


@Predictor.register("spd_predictor")
class SPDPredictor(Predictor):

    def predict(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        
        predict = self._model.vocab.get_token_from_index(
                                    np.argmax(output_dict["logits"], axis=-1),
                                    namespace="label")
        exps = list(np.exp(output_dict["logits"]))
        probs = [exps[0] / (exps[0] + exps[1]), exps[1] / (exps[0] + exps[1])]
        outputs = {
            "sentence": inputs["sentence"],
            "probs": {
                self._model.vocab.get_token_from_index(0, namespace="label"): probs[0],
                self._model.vocab.get_token_from_index(1, namespace="label"): probs[1]
            },
            "predict": predict
        }
        return outputs
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sent = json_dict["sentence"]  # type: str
        sentence = sent.strip().split(" ")
        instance = self._dataset_reader.text_to_instance(sentence=sentence)
        return instance