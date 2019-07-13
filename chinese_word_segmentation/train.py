import os
import logging

from allennlp.common import Params
from allennlp.common.util import prepare_global_logging, cleanup_global_logging, prepare_environment
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.training.trainer import Trainer

from assets import CWSDatasetReader, CWSModel


if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)


def train(config_path, serialization_dir):
    params = Params.from_file(config_path)
    stdout_handler = prepare_global_logging(serialization_dir, file_friendly_logging=False)
    prepare_environment(params)

    reader = DatasetReader.from_params(params.pop("dataset_reader", None))
    train_dataset = reader.read(params.pop("train_data_path", None))
    valid_dataset = reader.read(params.pop("validation_data_path", None))

    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    
    if not os.path.exists(serialization_dir):
        os.mkdir(serialization_dir)
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    # copy config file
    with open(config_path, "r", encoding="utf-8") as f_in:
        with open(os.path.join(serialization_dir, "config.json"),
                  "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
    
    model = Model.from_params(params.pop("model", None), vocab=vocab)

    iterator = DataIterator.from_params(params.pop("iterator", None))
    iterator.index_with(vocab)

    trainer = Trainer.from_params(model=model,
                                  serialization_dir=serialization_dir,
                                  iterator=iterator,
                                  train_data=train_dataset,
                                  validation_data=valid_dataset,
                                  params=params.pop("trainer", None))
    trainer.train()

    cleanup_global_logging(stdout_handler)


if __name__ == "__main__":
    config_path = "./config/config.json"
    serialization_dir = "./output/"
    train(config_path, serialization_dir)