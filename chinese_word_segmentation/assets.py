"""
This python module contains the implementation of (DatasetReader, Model, Predictor)
for Chinese Word Segmentation (CWS) task, based on the AllenNLP framework.
"""


import torch
import numpy as np

from typing import Iterator, List, Dict, Tuple
from overrides import overrides

from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import Predictor
from allennlp.common import JsonDict
from allennlp.data.dataset_readers.dataset_utils.span_utils import bmes_tags_to_spans


@DatasetReader.register("cws_dataset_reader")
class CWSDatasetReader(DatasetReader):

    def __init__(self) -> None:
        super().__init__(lazy=False)
    
    def text_to_instance(self,
                         sentence: List[str],
                         labels: List[str] = None) -> Instance:
        sentence_field = TextField([Token(token) for token in sentence],
                                   {"tokens": SingleIdTokenIndexer()})
        fields = {"sentence": sentence_field}
        if labels:
            seq_label_field = SequenceLabelField(labels=labels, sequence_field=sentence_field)
            fields["labels"] = seq_label_field
        return Instance(fields)
    
    def _parse_line(self, line: str) -> Tuple[List[str], List[str]]:
        """
        Given a line in `source.txt` (stripped), return the raw text (sentences)
        and its corresponding labels (B-word, M-word, E-word, S-word) for each
        character, i.e., this method will return a tuple: (raw_text, labels).

        If this line is not valid, it will return a tuple (None, None) instead.
        """
        after_split = line.split("  ")[1:]  # ignore the datetime info (position 0)
        if not after_split:
            return None, None
        tokens = []
        concat_flag = False
        concated_token = ""
        for item in after_split:
            item = item.strip()
            if not item:
                continue
            if not concat_flag:
                # named entity with single token
                if item[0] == "[" and "]" in item:
                    item = item[1:]
                    r_idx = item.rindex("]")
                    item = item[:r_idx]
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    tokens.append(item)
                elif item[0] == "[":
                    assert item[1] != "["
                    item = item[1:]
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    concated_token += item
                    concat_flag = True
                else:
                    assert "[" not in item
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    tokens.append(item)
            else:  # `concat_flag` is `True`
                # the data contains nested named entities, i.e., [..., [...]nt, ...]nt
                if "[" in item:
                    return None, None
                if "]" in item:
                    r_idx = item.rindex("]")
                    item = item[:r_idx]
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    tokens.append(concated_token + item)
                    concated_token = ""
                    concat_flag = False
                else:
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    concated_token += item
        assert not concated_token and not concat_flag
        if not tokens:
            return None, None
        raw_text, labels = [], []
        for token in tokens:
            if not token:  # data may contains some errors
                return None, None
            chars = [ch for ch in token]
            raw_text.extend(chars)
            if len(chars) == 1:
                labels.append("S-word")
            else:
                labels.extend(["B-word"] + ["M-word"] * (len(chars) - 2) + ["E-word"])
        return raw_text, labels

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:  # empty line
                    continue
                raw_text, labels = self._parse_line(line)
                if not raw_text:
                    continue
                yield self.text_to_instance(raw_text, labels)


@Model.register("cws_model")
class CWSModel(Model):
    """
    A allennlp model implementation for Chinese Word Segmentation (CWS).
    The model implementation follows Embeddings + Bi-LSTM + Linear architecture, 
    see `forward()` method for more details.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size("labels"))
        self.f1 = SpanBasedF1Measure(vocab, "labels", label_encoding="BMES")

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        
        if labels is not None:
            self.f1(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        matric = self.f1.get_metric(reset)
        return {"precision": matric["precision-overall"],
                "recall": matric["recall-overall"],
                "f1": matric["f1-measure-overall"]}


@Predictor.register("cws_predictor")
class CWSPredictor(Predictor):

    def predict(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        
        best_path = [self._model.vocab.get_token_from_index(index, namespace="labels")
                     for index in np.argmax(output_dict["tag_logits"], axis=-1)]

        sentence = inputs["sentence"]
        seg_sentence = []
        for _, span in bmes_tags_to_spans(best_path):
            span_start, span_end = span
            seg_sentence.append(sentence[span_start:span_end + 1])
        outputs = {
            "sentence": sentence,
            "seg_sentence": seg_sentence
        }
        return outputs
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sent = json_dict["sentence"]
        sentence = [ch for ch in sent]
        instance = self._dataset_reader.text_to_instance(sentence=sentence)
        return instance