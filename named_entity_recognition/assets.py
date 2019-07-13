"""
This python module contains the implementation of (DatasetReader, Model, Predictor)
for Named Entity Recognition (NER) task, based on the AllenNLP framework.
"""


import torch
import collections
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
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans


@DatasetReader.register("ner_dataset_reader")
class NERDatasetReader(DatasetReader):

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
        and its corresponding labels: 
            - NR: name of some person, e.g. 邓小平
            - NS: name of some place, e.g. 北京
            - NT: name of some institution or organization, e.g. 北京邮电大学
            - T: time, e.g. 一九九八年
        Note that the labels are in BIO favor, e.g.,
            一九八四年  ，  邓  小平  同志  来  到  深圳  视察  。
               B-T     O  B-NR I-NR   O   O   O   B-NS   O   O
        
        This method will return a tuple (tokens, labels). if this line is not valid, 
        it will return a tuple (None, None) instead.
        """
        after_split = line.split("  ")[1:]  # ignore the datetime info (position 0)
        if not after_split:
            return None, None
        tokens = []
        labels = []
        concat_flag = False
        concated_token = []
        for item in after_split:
            item = item.strip()
            if not item:
                continue
            if not concat_flag:
                # named entity with single token
                if item[0] == "[" and "]" in item:
                    item = item[1:]
                    r_idx = item.rindex("]")
                    label = item[r_idx + 1:].upper()
                    item = item[:r_idx]
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    tokens.append(item)
                    labels.append("B-" + label)
                elif item[0] == "[":
                    assert item[1] != "["
                    item = item[1:]
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    concated_token.append(item)
                    concat_flag = True
                else:
                    assert "[" not in item
                    r_idx = item.rindex("/")
                    label = item[r_idx + 1:].upper()
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    tokens.append(item)
                    if label == "NR" or label == "NRF":
                        labels.append("B-NR")
                    elif label == "NRG":
                        labels.append("I-NR")
                    elif label == "NS":
                        labels.append("B-NS")
                    elif label == "T":
                        labels.append("B-T")
                    else:
                        labels.append("O")
            else:  # `concat_flag` is `True`
                # the data contains nested named entities, i.e., [..., [...]nt, ...]nt
                if "[" in item:
                    return None, None
                if "]" in item:
                    r_idx = item.rindex("]")
                    label = item[r_idx + 1:].upper()
                    item = item[:r_idx]
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    concated_token.append(item)
                    tokens.extend(concated_token)
                    labels.extend(["B-" + label] + (len(concated_token) - 1) * ["I-" + label])
                    concated_token = []
                    concat_flag = False
                else:
                    r_idx = item.rindex("/")
                    item = item[:r_idx]
                    if "{" in item and "}" in item:
                        r_idx = item.rindex("{")
                        item = item[:r_idx]
                    concated_token.append(item)
        assert not concated_token and not concat_flag
        if not tokens:
            return None, None
        assert len(tokens) == len(labels)
        
        # merge continual B-xxx labels
        for i in range(len(tokens) - 1, 0, -1):
            if labels[i][0] != "B" or labels[i - 1][0] != "B":
                continue
            if labels[i][2:] == labels[i - 1][2:]:
                labels[i] = "I-" + labels[i][2:]
        
        char_level_tokens, char_level_labels = [], []
        for token, label in zip(tokens, labels):
            count = 0
            for ch in token:
                char_level_tokens.append(ch)
                count += 1
            if label == "O" or label.startswith("I-"):
                char_level_labels.extend([label] * count)
            else:
                assert label.startswith("B-")
                char_level_labels.extend([label] + (count - 1) * ["I-" + label[2:]])
        return char_level_tokens, char_level_labels


    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:  # empty line
                    continue
                tokens, labels = self._parse_line(line)
                if not tokens:
                    continue
                yield self.text_to_instance(tokens, labels)


@Model.register("ner_model")
class NERModel(Model):
    """
    A allennlp model implementation for Named Entity Recognition (NER).
    The model implementation follows Embeddings + Bi-LSTM + Linear architecture,
    see `forward()` method for details.
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
        self.f1 = SpanBasedF1Measure(vocab, "labels", label_encoding="BIO")

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


@Predictor.register("ner_predictor")
class NERPredictor(Predictor):

    def predict(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        
        predicted = [self._model.vocab.get_token_from_index(index, namespace="labels")
                     for index in np.argmax(output_dict["tag_logits"], axis=-1)]

        sentence = inputs["sentence"]
        named_entities = collections.defaultdict(list)
        for label, span in bio_tags_to_spans(predicted):
            span_start, span_end = span
            named_entities[label].append("".join(sentence[span_start:span_end + 1]))
        outputs = {
            "sentence": sentence,
            "named_entities": dict(named_entities)
        }
        return outputs
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sent = json_dict["sentence"]
        sentence = [ch for ch in sent]
        instance = self._dataset_reader.text_to_instance(sentence=sentence)
        return instance